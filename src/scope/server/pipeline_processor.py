"""Pipeline processor for running a single pipeline in a thread."""

import logging
import queue
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import torch

from scope.core.pipelines.controller import parse_ctrl_input

from .kafka_publisher import publish_event
from .media_packets import (
    AudioPacket,
    MediaTimestamp,
    VideoPacket,
    ensure_video_packet,
)
from .pipeline_manager import PipelineNotAvailableException
from .tempo_sync import get_beat_boundary

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 2

SLEEP_TIME = 0.01

# Sentinel sample_rate value used to signal the audio track to flush its buffer.
# Sent as (None, AUDIO_FLUSH_SENTINEL) through the audio_output_queue.
AUDIO_FLUSH_SENTINEL = -1

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
BATCH_FPS_SAMPLE_SIZE = 10  # Number of batch-level samples for windowed averaging


class PipelineProcessor:
    """Processes frames through a single pipeline in a dedicated thread."""

    def __init__(
        self,
        pipeline: Any,
        pipeline_id: str,
        initial_parameters: dict = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
        tempo_sync: Any | None = None,
        modulation_engine: Any | None = None,
        node_id: str | None = None,
        notification_callback: Callable[[dict], None] | None = None,
    ):
        """Initialize a pipeline processor.

        Args:
            pipeline: Pipeline instance to process frames with
            pipeline_id: ID of the pipeline (used for logging)
            initial_parameters: Initial parameters for the pipeline
            session_id: Session ID for event tracking
            user_id: User ID for event tracking
            connection_id: Connection ID from fal.ai WebSocket for event correlation
            connection_info: Connection metadata (gpu_type, region, etc.)
            tempo_sync: TempoSync instance for beat state injection
            modulation_engine: ModulationEngine for beat-synced param modulation
            node_id: Graph node ID (used for per-node parameter routing in graph mode)
            notification_callback: Lets consumers know of parameter updates etc
        """
        self.pipeline = pipeline
        self.pipeline_id = pipeline_id
        self.node_id = node_id or pipeline_id
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        self.tempo_sync = tempo_sync
        self.modulation_engine = modulation_engine
        self.notification_callback = notification_callback

        # Port-based queues wired by graph_executor.build_graph()
        self.input_queues: dict[str, queue.Queue] = {}
        self.output_queues: dict[str, list[queue.Queue]] = {}
        # Lock to protect input_queues assignment for thread-safe reference swapping
        self.input_queue_lock = threading.Lock()
        # External dict references that hold output queues (e.g. sink_queues_by_node,
        # record_queues_by_node). Updated by _resize_output_queue so cached
        # references stay in sync when a queue object is replaced.
        self.external_queue_refs: list[tuple[dict, str]] = []

        # Audio output queue: (audio_tensor, sample_rate) tuples.
        # Consumed by FrameProcessor.get_audio() on the sink processor.
        # Flushed on prompt change, so only needs enough headroom for
        # bursty production (pipeline thread outpacing real-time playback).
        self.audio_output_queue: queue.Queue[AudioPacket | tuple[torch.Tensor, int]] = (
            queue.Queue(maxsize=10)
        )

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=8)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Output FPS tracking (batch-level throughput)
        # Stores (num_frames, interval) tuples so that FPS = sum(frames) / sum(intervals),
        # correctly handling variable batch sizes across pipeline calls
        self._batch_samples: deque[tuple[int, float]] = deque(
            maxlen=BATCH_FPS_SAMPLE_SIZE
        )
        self._last_batch_time: float | None = None
        # Start with a higher initial FPS to prevent initial queue buildup
        self.current_output_fps = MAX_FPS
        self.output_fps_lock = threading.Lock()

        self.paused = False
        # Input mode is signaled by the frontend at stream start
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Maps output port -> list of (consumer_processor, consumer_input_port).
        # Used by _resize_output_queue to update all downstream consumers when
        # a queue is replaced. Populated by graph_executor.build_graph.
        self.output_consumers: dict[str, list[tuple[PipelineProcessor, str]]] = {}

        # Flag to track pending cache initialization after queue flush
        # Set when reset_cache flushes queues, cleared after successful pipeline call
        self._pending_cache_init = False

        # Beat-synced cache reset: fire init_cache=True at rhythmic intervals
        self._beat_cache_reset_rate: str = "none"
        self._last_reset_boundary: int = -1

        # Native frame rate reported by the pipeline (e.g. 24fps for LTX-2).
        # When set, get_fps() returns this instead of the measured production rate,
        # giving the video track a stable playback speed for A/V sync.
        self.native_fps: float | None = None

        # Names of input ports whose declared type isn't "video". Values
        # arriving on these ports are drained into ``self.parameters`` each
        # chunk instead of being treated as chunked video streams.
        self._non_video_input_ports: set[str] = {
            p.name
            for p in self.pipeline.get_definition().inputs
            if p.port_type != "video"
        }

    def _drain_non_video_inputs(self) -> None:
        """Drain scalar input queues into ``self.parameters`` and sync the UI.

        Non-video ports (string, number, …) deliver discrete values rather
        than frame streams; the latest value on each queue wins. Video ports
        are left alone — those follow the chunk-gathering path below. Drained
        values are emitted via ``self.notification_callback`` so widgets like
        the Prompt textarea reflect what an upstream backend node produced.
        Where the notification ends up (WebRTC data channel, events trickle
        channel, …) is the constructor caller's concern.
        """
        if not self._non_video_input_ports:
            return
        with self.input_queue_lock:
            targets = [
                (port, q)
                for port, q in self.input_queues.items()
                if port in self._non_video_input_ports
            ]
        drained_values: dict[str, Any] = {}
        for port, q in targets:
            latest = None
            drained = False
            while True:
                try:
                    latest = q.get_nowait()
                    drained = True
                except queue.Empty:
                    break
            if drained:
                self.parameters[port] = latest
                drained_values[port] = latest
        if not drained_values:
            return
        if self.notification_callback is None:
            return
        payload = {"node_id": self.node_id, **drained_values}
        try:
            self.notification_callback(
                {"type": "parameters_updated", "parameters": payload}
            )
        except Exception:
            logger.debug(
                "Failed to notify parameters_updated for %s",
                self.node_id,
                exc_info=True,
            )

    def set_beat_cache_reset_rate(self, rate: str) -> None:
        """Set the beat-synced cache reset rate and reset the boundary tracker."""
        self._beat_cache_reset_rate = rate
        self._last_reset_boundary = -1

    def _resize_output_queue(self, port: str, target_size: int):
        """Resize output queues for a given port, transferring existing frames.

        Handles fan-out (multiple queues per port) and port name remapping
        (output port name may differ from consumer's input port name).
        Consumer references are updated via output_consumers which is populated
        by graph_executor.build_graph.
        """
        port_queues = self.output_queues.get(port)
        if not port_queues:
            return

        consumers = self.output_consumers.get(port, [])
        new_list = []
        resized = False

        for old_q in port_queues:
            if old_q.maxsize >= target_size:
                new_list.append(old_q)
                continue

            logger.info(
                f"Increasing output queue size for port '{port}' to {target_size}, "
                f"current size {old_q.maxsize}"
            )
            new_q = queue.Queue(maxsize=target_size)
            while not old_q.empty():
                try:
                    frame = old_q.get_nowait()
                    new_q.put_nowait(frame)
                except queue.Empty:
                    break
            new_list.append(new_q)
            resized = True

            # Update every consumer whose input queue is the old queue object
            for consumer, consumer_port in consumers:
                with consumer.input_queue_lock:
                    if consumer.input_queues.get(consumer_port) is old_q:
                        consumer.input_queues[consumer_port] = new_q

            # Update external references (sink/record queues in SinkManager)
            for ref_dict, ref_key in self.external_queue_refs:
                if ref_dict.get(ref_key) is old_q:
                    ref_dict[ref_key] = new_q

        if resized:
            self.output_queues[port] = new_list

    @property
    def output_queue(self) -> queue.Queue | None:
        """Primary video output queue (used by sink to read frames)."""
        queues = self.output_queues.get("video")
        return queues[0] if queues else None

    def start(self):
        """Start the pipeline processor thread."""
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info(f"PipelineProcessor started for pipeline: {self.pipeline_id}")

    def stop(self):
        """Stop the pipeline processor thread."""
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        # Clear all input queues
        with self.input_queue_lock:
            input_queues_copy = dict(self.input_queues)
        for q in input_queues_copy.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        for queues in self.output_queues.values():
            for q in queues:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

        logger.info(f"PipelineProcessor stopped for pipeline: {self.pipeline_id}")

    def _flush_audio(self):
        """Drain the audio output queue and send a flush sentinel.

        Called when prompts change so the audio track discards buffered
        audio from the previous prompt and plays the new speech immediately.
        """
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.audio_output_queue.put_nowait((None, AUDIO_FLUSH_SENTINEL))
        except queue.Full:
            pass

    def _should_flush_audio_on_prompt_change(self) -> bool:
        """Only flush prompt audio when it cannot desync queued video."""
        return not self.output_queues.get("video")

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.info(
                f"Parameter queue full for {self.pipeline_id}, dropping parameter update"
            )
            return False

    def worker_loop(self):
        """Main worker loop that processes frames."""
        logger.info(f"Worker thread started for pipeline: {self.pipeline_id}")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except PipelineNotAvailableException as e:
                logger.debug(
                    f"Pipeline {self.pipeline_id} temporarily unavailable: {e}"
                )
                # Sleep briefly and continue
                self.shutdown_event.wait(SLEEP_TIME)
                continue
            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(
                        f"Error in worker loop for {self.pipeline_id}: {e}",
                        exc_info=True,
                    )
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop for {self.pipeline_id}: {e}, stopping"
                    )
                    # Publish error event for pipeline processing failure
                    publish_event(
                        event_type="error",
                        session_id=self.session_id,
                        connection_id=self.connection_id,
                        pipeline_ids=[self.pipeline_id],
                        user_id=self.user_id,
                        error={
                            "error_type": "pipeline_processing_failed",
                            "message": str(e),
                            "exception_type": type(e).__name__,
                            "recoverable": False,
                        },
                        connection_info=self.connection_info,
                    )
                    break

        logger.info(f"Worker thread stopped for pipeline: {self.pipeline_id}")

    def prepare_chunk(
        self, input_queue_ref: queue.Queue, chunk_size: int
    ) -> list[VideoPacket]:
        """
        Sample frames uniformly from one queue (used when only video port is present).
        """
        step = input_queue_ref.qsize() / chunk_size
        indices = [round(i * step) for i in range(chunk_size)]
        video_frames: list[VideoPacket] = []
        last_idx = indices[-1]
        for i in range(last_idx + 1):
            frame = ensure_video_packet(input_queue_ref.get_nowait())
            if i in indices:
                video_frames.append(frame)
        return video_frames

    def prepare_multi_chunk(
        self,
        input_queues_ref: dict[str, queue.Queue],
        chunk_size: int,
    ) -> dict[str, list[VideoPacket]]:
        """
        Sample chunk_size frames uniformly from each wired queue.

        All queues must have >= chunk_size frames (caller checks readiness).
        Each port is sampled independently using the same uniform strategy.
        """
        return {
            port: self.prepare_chunk(q, chunk_size)
            for port, q in input_queues_ref.items()
        }

    @staticmethod
    def _normalize_timestamps(
        raw_timestamps: Any, expected_len: int
    ) -> list[MediaTimestamp]:
        if not isinstance(raw_timestamps, list):
            return [MediaTimestamp() for _ in range(expected_len)]

        normalized: list[MediaTimestamp] = []
        for ts in raw_timestamps[:expected_len]:
            if isinstance(ts, MediaTimestamp):
                normalized.append(ts)
                continue
            if isinstance(ts, dict):
                pts = ts.get("pts")
                time_base = ts.get("time_base")
                if time_base is not None and not isinstance(time_base, Fraction):
                    try:
                        time_base = Fraction(time_base)
                    except Exception:
                        time_base = None
                normalized.append(MediaTimestamp(pts=pts, time_base=time_base))
                continue
            normalized.append(MediaTimestamp())

        if len(normalized) < expected_len:
            normalized.extend(
                MediaTimestamp() for _ in range(expected_len - len(normalized))
            )
        return normalized

    def process_chunk(self):
        """Process a single chunk of frames."""
        # Check if there are new parameters
        try:
            new_parameters = self.parameters_queue.get_nowait()
            if new_parameters != self.parameters:
                # Flush stale audio for audio-only pipelines so the new speech
                # starts immediately. For A/V pipelines, keep audio queued:
                # video frames already emitted for the previous prompt remain
                # queued as well, and dropping only audio makes them play silent.
                if "prompts" in new_parameters and new_parameters.get(
                    "prompts"
                ) != self.parameters.get("prompts"):
                    if self._should_flush_audio_on_prompt_change():
                        self._flush_audio()

                # Clear stale transition when new prompts arrive without transition
                if (
                    "prompts" in new_parameters
                    and "transition" not in new_parameters
                    and "transition" in self.parameters
                ):
                    self.parameters.pop("transition", None)

                # Update video mode if input_mode parameter changes
                if "input_mode" in new_parameters:
                    self._video_mode = new_parameters.get("input_mode") == "video"

                # Accumulate ctrl_input: keys = latest, mouse = sum
                if "ctrl_input" in new_parameters:
                    if "ctrl_input" in self.parameters:
                        existing = self.parameters["ctrl_input"]
                        new_ctrl = new_parameters["ctrl_input"]
                        new_parameters["ctrl_input"] = {
                            "button": new_ctrl.get("button", []),
                            "mouse": [
                                existing.get("mouse", [0, 0])[0]
                                + new_ctrl.get("mouse", [0, 0])[0],
                                existing.get("mouse", [0, 0])[1]
                                + new_ctrl.get("mouse", [0, 0])[1],
                            ],
                        }

                # Merge new parameters with existing ones
                self.parameters = {**self.parameters, **new_parameters}
        except queue.Empty:
            pass

        # Pause or resume the processing
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            # Reset so the next batch FPS sample doesn't span the pause/unpause gap
            self._last_batch_time = None
            self.paused = paused
        if self.paused:
            # While paused, a one-shot `step: true` trigger advances the
            # pipeline by exactly one chunk before falling back to paused.
            # Used by frame-by-frame render drivers that clock the pipeline
            # off-line — engines stay loaded, no separate session needed.
            if self.parameters.pop("step", False):
                pass  # fall through to process_chunk
            else:
                self.shutdown_event.wait(SLEEP_TIME)
                return

        # Prepare pipeline
        reset_cache = self.parameters.pop("reset_cache", None)
        lora_scales = self.parameters.pop("lora_scales", None)

        # Handle reset_cache: clear this processor's output queues
        if reset_cache:
            logger.info(f"Clearing cache for pipeline processor: {self.pipeline_id}")
            for queues in self.output_queues.values():
                for q in queues:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
            self._pending_cache_init = True

        # Drain non-video input ports (string, number, …) into parameters so
        # upstream nodes — e.g. a PromptEnhancer feeding a string port — can
        # drive pipeline kwargs via graph edges.
        self._drain_non_video_inputs()

        requirements = None
        if hasattr(self.pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected
                prepare_params["video"] = True
            requirements = self.pipeline.prepare(**prepare_params)

        chunks: dict[str, list[VideoPacket]] = {}
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.input_queue_lock:
                input_queues_ref = {
                    port: q
                    for port, q in self.input_queues.items()
                    if port not in self._non_video_input_ports
                }
            # Wait until ALL wired video input queues have enough frames
            if not input_queues_ref or not all(
                q.qsize() >= current_chunk_size for q in input_queues_ref.values()
            ):
                # Preserve popped one-shot parameters so they are applied once frames arrive
                if lora_scales is not None:
                    self.parameters["lora_scales"] = lora_scales
                self.shutdown_event.wait(SLEEP_TIME)
                return
            if len(input_queues_ref) == 1:
                port, q = next(iter(input_queues_ref.items()))
                chunks[port] = self.prepare_chunk(q, current_chunk_size)
            else:
                chunks = self.prepare_multi_chunk(input_queues_ref, current_chunk_size)

        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Clear one-shot parameters from self.parameters now that they are captured
            # in call_params. Popping before pipeline execution (instead of after success)
            # ensures a failure — e.g. a bad image URL that raises FileNotFoundError —
            # does not re-fire the same value on every subsequent chunk, which previously
            # produced thousands of repeated tracebacks per second.
            one_shot_params = (
                "vace_ref_images",
                "images",
                "first_frame_image",
                "last_frame_image",
            )
            for param in one_shot_params:
                self.parameters.pop(param, None)

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared or self._pending_cache_init
            if reset_cache:
                call_params["init_cache"] = True

            # Pass lora_scales only when present
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Extract ctrl_input, parse it, and reset mouse for next frame
            if "ctrl_input" in self.parameters:
                ctrl_data = self.parameters["ctrl_input"]
                call_params["ctrl_input"] = parse_ctrl_input(ctrl_data)
                # Reset mouse accumulator, keep key state
                self.parameters["ctrl_input"]["mouse"] = [0.0, 0.0]

            # Fill call_params from stream chunks (port names are set by graph edges)
            if chunks:
                for port, packet_list in chunks.items():
                    call_params[port] = [packet.tensor for packet in packet_list]
                    ts_key = (
                        "video_timestamps" if port == "video" else f"{port}_timestamps"
                    )
                    call_params[ts_key] = [packet.timestamp for packet in packet_list]

            if self.tempo_sync is not None:
                call_params = self._apply_tempo_sync(call_params)

            processing_start = time.time()
            output_dict = self.pipeline(**call_params)
            processing_time = time.time() - processing_start

            if not output_dict:
                # 1) Some pipelines return {} when idle
                # 2) For those, prepare() is None, so we never wait on input queues.
                # 3) Without this sleep the worker thread would busy-loop.
                self.shutdown_event.wait(SLEEP_TIME)
                return

            # Pass audio to output queue regardless of whether video exists.
            # This ensures audio-only pipelines can deliver audio.
            audio_output = output_dict.get("audio")
            audio_sample_rate = output_dict.get("audio_sample_rate")
            if audio_output is not None and audio_sample_rate is not None:
                try:
                    audio_cpu = audio_output.detach().cpu()
                    audio_ts = output_dict.get("audio_timestamps")
                    timestamp = MediaTimestamp()
                    if isinstance(audio_ts, list) and audio_ts:
                        first = audio_ts[0]
                        if isinstance(first, MediaTimestamp):
                            timestamp = first
                    elif isinstance(audio_ts, MediaTimestamp):
                        timestamp = audio_ts
                    self.audio_output_queue.put_nowait(
                        AudioPacket(
                            audio=audio_cpu,
                            sample_rate=audio_sample_rate,
                            timestamp=timestamp,
                        )
                    )
                except queue.Full:
                    logger.warning(
                        "Audio output queue full for %s, dropping audio chunk",
                        self.pipeline_id,
                    )

            # Extract video from the returned dictionary
            output = output_dict.get("video")
            if output is None:
                self.is_prepared = True
                self._pending_cache_init = False
                return

            # Clear transition when complete
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(self.pipeline, "state"):
                    transition_active = self.pipeline.state.get(
                        "_transition_active", False
                    )

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    self.parameters.pop("transition", None)

            num_frames = 0
            if output is not None:
                num_frames = output.shape[0]

            # Put each output port's frames to its queues (all frame ports are streamed)
            for port, value in output_dict.items():
                if value is None or not isinstance(value, torch.Tensor):
                    continue
                queues = self.output_queues.get(port)
                if not queues:
                    continue
                # Convert batch-format tensors [B, C, F, H, W] to frame format
                # [B*F, H, W, C] so they can be split and queued as individual
                # frames.  This handles outputs from preprocessor pipelines
                # (e.g. VACE frames/masks) which produce pre-batched 5D tensors
                # rather than per-frame 4D tensors.
                if value.dim() == 5:
                    b, c, f, h, w = value.shape
                    value = (
                        value.permute(0, 2, 3, 4, 1)
                        .reshape(b * f, h, w, c)
                        .contiguous()
                    )
                    # Convert [-1, 1] to [0, 1] for uint8 encoding; downstream
                    # preprocess_chunk reverses this via (x / 255) * 2 - 1.
                    if value.min() < 0:
                        value = (value + 1.0) / 2.0
                # Resize output queues to fit at least one full batch
                target_size = value.shape[0] * OUTPUT_QUEUE_MAX_SIZE_FACTOR
                self._resize_output_queue(port, target_size)
                # Re-read queues after potential resize – _resize_output_queue
                # may replace self.output_queues[port] with a new list.
                queues = self.output_queues.get(port)
                if not queues:
                    continue
                if value.dtype != torch.uint8:
                    value = (
                        (value * 255.0)
                        .clamp(0, 255)
                        .to(dtype=torch.uint8)
                        .contiguous()
                        .detach()
                    )
                frames = [value[i].unsqueeze(0) for i in range(value.shape[0])]
                ts_key = f"{port}_timestamps"
                raw_timestamps = output_dict.get(ts_key)
                if port == "video" and raw_timestamps is None:
                    raw_timestamps = output_dict.get("video_timestamps")
                timestamps = self._normalize_timestamps(raw_timestamps, len(frames))
                for idx, frame in enumerate(frames):
                    packet = VideoPacket(tensor=frame, timestamp=timestamps[idx])
                    for q in queues:
                        try:
                            if q is queues[0]:
                                q.put_nowait(packet)
                            else:
                                q.put_nowait(
                                    VideoPacket(
                                        tensor=packet.tensor.clone(),
                                        timestamp=packet.timestamp,
                                    )
                                )
                        except queue.Full:
                            logger.debug(
                                f"Output queue full for {self.pipeline_id} port '{port}', dropping frame"
                            )

            # Latch native frame rate for stable playback speed.
            # Check output dict first, then pipeline config as fallback.
            frame_rate = output_dict.get("frame_rate")
            if frame_rate is None and hasattr(self.pipeline, "config"):
                frame_rate = getattr(self.pipeline.config, "frame_rate", None)
            if frame_rate is not None and float(frame_rate) > 0:
                self.native_fps = float(frame_rate)

            # Track batch-level throughput for FPS calculation
            if output is not None and num_frames > 0:
                self._track_output_batch(num_frames, processing_time)

            # Forward extra params (non-video outputs without queues) to downstream
            # pipelines. Preprocessors may return e.g. {"video": frames,
            # "vace_input_frames": ..., "vace_input_masks": ...} and the extra
            # entries need to reach the consuming pipeline as parameters.
            extra_params = {
                k: v
                for k, v in output_dict.items()
                if k not in self.output_queues
                and k not in {"video_timestamps", "audio_timestamps"}
                and not k.endswith("_timestamps")
            }
            if extra_params and self.output_consumers:
                seen: set[int] = set()
                for consumers in self.output_consumers.values():
                    for consumer_proc, _ in consumers:
                        proc_id = id(consumer_proc)
                        if proc_id not in seen:
                            seen.add(proc_id)
                            consumer_proc.update_parameters(extra_params)

        except Exception as e:
            if self._is_recoverable(e):
                logger.error(
                    f"Error processing chunk for {self.pipeline_id}: {e}", exc_info=True
                )
            else:
                raise e

        self.is_prepared = True
        self._pending_cache_init = False

    def _apply_tempo_sync(self, call_params: dict) -> dict:
        """Inject beat state, apply modulation, and handle beat-synced cache resets."""
        beat_state = self.tempo_sync.get_beat_state()
        if beat_state is None:
            return call_params

        call_params["bpm"] = beat_state.bpm
        call_params["beat_phase"] = beat_state.beat_phase
        call_params["bar_position"] = beat_state.bar_position
        call_params["beat_count"] = beat_state.beat_count
        call_params["is_playing"] = beat_state.is_playing

        if self.modulation_engine is not None:
            call_params = self.modulation_engine.apply(
                beat_phase=beat_state.beat_phase,
                bar_position=beat_state.bar_position,
                beat_count=beat_state.beat_count,
                beats_per_bar=self.tempo_sync.beats_per_bar,
                params=call_params,
            )

        if self._beat_cache_reset_rate != "none":
            boundary = get_beat_boundary(
                self._beat_cache_reset_rate,
                beat_state.beat_count,
                self.tempo_sync.beats_per_bar,
            )
            if boundary != self._last_reset_boundary and self._last_reset_boundary >= 0:
                call_params["init_cache"] = True
                call_params["base_seed"] = random.randint(0, 2**31 - 1)
                logger.info(
                    "[BEAT RESET] Cache reset + seed change at boundary %d (rate=%s)",
                    boundary,
                    self._beat_cache_reset_rate,
                )
            self._last_reset_boundary = boundary

        return call_params

    def _track_output_batch(self, num_frames: int, processing_time: float):
        """Track batch-level production throughput for FPS calculation.

        Stores (num_frames, interval) tuples and computes FPS as
        sum(frames) / sum(intervals). This correctly handles variable
        batch sizes and avoids the oscillation caused by per-frame delta
        tracking where near-zero intra-batch deltas mixed with large
        inter-batch gaps cause the FPS estimate to swing permanently.

        On the first call, processing_time is used as the interval since
        there is no previous batch to measure against. This gives a useful
        FPS estimate immediately rather than waiting for a second batch.
        """
        now = time.time()
        with self.output_fps_lock:
            if self._last_batch_time is not None:
                interval = now - self._last_batch_time
            elif processing_time > 0:
                # First batch: use processing time as initial interval estimate
                interval = processing_time
            else:
                interval = 0

            if interval > 0:
                self._batch_samples.append((num_frames, interval))

            self._last_batch_time = now

        self._calculate_output_fps()

    def _calculate_output_fps(self):
        """Calculate FPS from batch-level throughput: sum(frames) / sum(intervals)."""
        with self.output_fps_lock:
            if self._batch_samples:
                total_frames = sum(n for n, _ in self._batch_samples)
                total_time = sum(t for _, t in self._batch_samples)
                if total_time > 0:
                    fps = total_frames / total_time
                    self.current_output_fps = max(MIN_FPS, min(MAX_FPS, fps))

    def get_fps(self) -> float:
        """Get the playback FPS for this pipeline's output.

        If the pipeline reports a native frame rate (e.g. 24fps for LTX-2),
        that value is returned for stable playback. Otherwise falls back to
        the measured production rate.
        """
        if self.native_fps is not None:
            return self.native_fps
        with self.output_fps_lock:
            output_fps = self.current_output_fps
        return min(MAX_FPS, output_fps)

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        return True
