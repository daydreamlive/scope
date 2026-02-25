"""Pipeline processor for running a single pipeline in a thread."""

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import torch

from scope.core.pipelines.controller import parse_ctrl_input
from scope.core.pipelines.wan2_1.vace import VACEEnabledPipeline

from .kafka_publisher import publish_event
from .pipeline_manager import PipelineNotAvailableException
from .pipeline_throttler import PipelineThrottler

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

SLEEP_TIME = 0.01

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
OUTPUT_FPS_SAMPLE_SIZE = 30
OUTPUT_FPS_MIN_SAMPLES = 2


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
        node_id: str | None = None,
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
            node_id: Graph node ID (used for per-node parameter routing in graph mode)
        """
        self.pipeline = pipeline
        self.pipeline_id = pipeline_id
        self.node_id = node_id or pipeline_id
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info

        # Unified port-based queues: all frame streams (video, vace_input_frames, etc.) use queues
        self.input_queues: dict[str, queue.Queue] = {
            "video": queue.Queue(maxsize=30),
        }
        self.output_queues: dict[str, list[queue.Queue]] = {
            "video": [queue.Queue(maxsize=8)],
        }
        # Lock to protect input_queues assignment for thread-safe reference swapping
        self.input_queue_lock = threading.Lock()

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=8)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Output FPS tracking (based on frames added to output queue)
        # Stores inter-frame durations (seconds)
        self.output_frame_deltas = deque(maxlen=OUTPUT_FPS_SAMPLE_SIZE)
        self._last_frame_time: float | None = None
        # Start with a higher initial FPS to prevent initial queue buildup
        self.current_output_fps = MAX_FPS
        self.output_fps_lock = threading.Lock()

        self.paused = False
        # Input mode is signaled by the frontend at stream start
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Reference to next processor in chain (if chained)
        self.next_processor: PipelineProcessor | None = None

        # Maps output port -> list of (consumer_processor, consumer_input_port).
        # Used by _resize_output_queue to update all downstream consumers when
        # a queue is replaced.  Populated by set_next_processor (chain mode) and
        # by graph_executor.build_graph (graph mode, supports port remapping).
        self.output_consumers: dict[str, list[tuple[PipelineProcessor, str]]] = {}

        # Route based on frontend's VACE intent (not pipeline.vace_enabled which is lazy-loaded)
        # This fixes the chicken-and-egg problem where VACE isn't enabled until vace_input_frames arrives
        self.vace_enabled = (initial_parameters or {}).get("vace_enabled", False)
        self.vace_use_input_video = (initial_parameters or {}).get(
            "vace_use_input_video", True
        )

        # Cache VACE support check to avoid isinstance on every chunk
        self._pipeline_supports_vace = isinstance(pipeline, VACEEnabledPipeline)

        # Flag to track pending cache initialization after queue flush
        # Set when reset_cache flushes queues, cleared after successful pipeline call
        self._pending_cache_init = False

        # Throttler for controlling processing rate in chained pipelines
        # Throttling is applied when this pipeline produces frames faster than
        # the next pipeline in the chain can consume them
        self.throttler = PipelineThrottler()

    def _resize_output_queue(self, port: str, target_size: int):
        """Resize output queues for a given port, transferring existing frames.

        Handles fan-out (multiple queues per port) and port name remapping
        (output port name may differ from consumer's input port name).
        Consumer references are updated via output_consumers which is populated
        by set_next_processor (chain mode) or graph_executor (graph mode).
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

        if resized:
            self.output_queues[port] = new_list

    def set_next_processor(self, next_processor: "PipelineProcessor"):
        """Set the next processor in the chain and update output queue size accordingly.

        Args:
            next_processor: The next pipeline processor in the chain
        """
        self.next_processor = next_processor

        # Register as output consumer for queue resize updates
        self.output_consumers.setdefault("video", []).append((next_processor, "video"))

        # Set throttler's reference to next processor for throttling decisions
        self.throttler.set_next_processor(next_processor)

        # Calculate output queue size based on next processor's requirements
        next_pipeline = next_processor.pipeline
        if hasattr(next_pipeline, "prepare"):
            requirements = next_pipeline.prepare(video=True)
            input_size = requirements.input_size
            target_size = max(8, input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR)
            self._resize_output_queue("video", target_size)

        # Update next processor's input_queue to point to this output_queue
        # Use lock to ensure thread-safe reference swapping
        with next_processor.input_queue_lock:
            next_processor.input_queues["video"] = self.output_queues["video"][0]

    @property
    def input_queue(self) -> queue.Queue | None:
        """Primary video input queue (chain mode, get_fps, resize)."""
        return self.input_queues.get("video")

    @property
    def output_queue(self) -> queue.Queue | None:
        """Primary video output queue (for chain mode and sink get())."""
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
        self.throttler.interrupt()

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
    ) -> list[torch.Tensor]:
        """
        Sample frames uniformly from one queue (used when only video port is present).
        """
        step = input_queue_ref.qsize() / chunk_size
        indices = [round(i * step) for i in range(chunk_size)]
        video_frames = []
        last_idx = indices[-1]
        for i in range(last_idx + 1):
            frame = input_queue_ref.get_nowait()
            if i in indices:
                video_frames.append(frame)
        return video_frames

    def prepare_multi_chunk(
        self,
        input_queues_ref: dict[str, queue.Queue],
        chunk_size: int,
    ) -> dict[str, list[torch.Tensor]]:
        """
        Sample chunk_size frames uniformly from each wired queue.

        All queues must have >= chunk_size frames (caller checks readiness).
        Each port is sampled independently using the same uniform strategy.
        """
        return {
            port: self.prepare_chunk(q, chunk_size)
            for port, q in input_queues_ref.items()
        }

    def process_chunk(self):
        """Process a single chunk of frames."""
        # Check if there are new parameters
        try:
            new_parameters = self.parameters_queue.get_nowait()
            if new_parameters != self.parameters:
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
            # Reset so the next FPS delta doesn't span the pause/unpause gap
            self._last_frame_time = None
            self.paused = paused
        if self.paused:
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

        requirements = None
        if hasattr(self.pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected
                prepare_params["video"] = True
            requirements = self.pipeline.prepare(**prepare_params)

        chunks: dict[str, list[torch.Tensor]] = {}
        input_frame_count = 0
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.input_queue_lock:
                input_queues_ref = dict(self.input_queues)
            # Wait until ALL wired input queues have enough frames
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
            input_frame_count = max(
                (len(frames) for frames in chunks.values()), default=0
            )

        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())
            if not self.is_prepared:
                logger.info(
                    f"[DEBUG] First call for {self.pipeline_id}: "
                    f"params keys={sorted(self.parameters.keys())}, "
                    f"has_prompts={'prompts' in self.parameters}"
                )

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared or self._pending_cache_init
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass lora_scales only when present
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Extract ctrl_input, parse it, and reset mouse for next frame
            if "ctrl_input" in self.parameters:
                ctrl_data = self.parameters["ctrl_input"]
                call_params["ctrl_input"] = parse_ctrl_input(ctrl_data)
                # Reset mouse accumulator, keep key state
                self.parameters["ctrl_input"]["mouse"] = [0.0, 0.0]

            # Fill call_params from stream chunks
            if chunks:
                # Handle VACE inputs from graph edges (each port independently)
                if chunks.get("vace_input_frames"):
                    call_params["vace_input_frames"] = chunks["vace_input_frames"]
                if chunks.get("vace_input_masks"):
                    call_params["vace_input_masks"] = chunks["vace_input_masks"]
                if chunks.get("video"):
                    if (
                        self._pipeline_supports_vace
                        and self.vace_enabled
                        and self.vace_use_input_video
                        and "vace_input_frames" not in call_params
                    ):
                        call_params["vace_input_frames"] = chunks["video"]
                    else:
                        call_params["video"] = chunks["video"]
                # Pass any other stream ports (e.g. video2 for combine_streams)
                for port, frame_list in chunks.items():
                    if port in ("video", "vace_input_frames", "vace_input_masks"):
                        continue
                    call_params[port] = frame_list

            processing_start = time.time()
            output_dict = self.pipeline(**call_params)
            processing_time = time.time() - processing_start

            output = output_dict.get("video")
            if output is None:
                return

            # Clear one-shot parameters after use to prevent sending them on subsequent chunks
            # These parameters should only be sent when explicitly provided in parameter updates
            one_shot_params = [
                "vace_ref_images",
                "images",
                "first_frame_image",
                "last_frame_image",
            ]
            for param in one_shot_params:
                if param in call_params and param in self.parameters:
                    self.parameters.pop(param, None)

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

            num_frames = output.shape[0]

            # Record batch timing for throttling calculations
            if input_frame_count > 0:
                self.throttler.record_input_batch(input_frame_count, processing_time)
            if num_frames > 0:
                self.throttler.record_output_batch(num_frames, processing_time)

            # Normalize to [0, 255] and convert to uint8
            # Keep frames on GPU - frame_processor handles CPU transfer for streaming
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
            )

            # Put each output port's frames to its queues (all frame ports are streamed)
            for port, value in output_dict.items():
                if value is None or not isinstance(value, torch.Tensor):
                    continue
                queues = self.output_queues.get(port)
                if not queues:
                    continue
                # Resize output queues to meet target max size.
                # Only resize when there are downstream pipeline consumers;
                # the sink has no consumers so its queues stay fixed for
                # frame_processor.get().
                if self.output_consumers.get(port):
                    target_size = value.shape[0] * OUTPUT_QUEUE_MAX_SIZE_FACTOR
                    self._resize_output_queue(port, target_size)
                if value.dtype != torch.uint8:
                    value = (
                        (value * 255.0)
                        .clamp(0, 255)
                        .to(dtype=torch.uint8)
                        .contiguous()
                        .detach()
                    )
                frames = [value[i].unsqueeze(0) for i in range(value.shape[0])]
                for frame in frames:
                    if port == "video":
                        self._track_output_frame()
                    for q in queues:
                        try:
                            q.put_nowait(frame if q is queues[0] else frame.clone())
                        except queue.Full:
                            if port == "video":
                                logger.info(
                                    f"Output queue full for {self.pipeline_id}, dropping frame"
                                )
                            break

            if chunks and self.next_processor is not None:
                self.throttler.throttle()

        except Exception as e:
            if self._is_recoverable(e):
                logger.error(
                    f"Error processing chunk for {self.pipeline_id}: {e}", exc_info=True
                )
            else:
                raise e

        self.is_prepared = True
        self._pending_cache_init = False

    def _track_output_frame(self):
        """Track when a frame is added to the output queue (production rate).

        Stores inter-frame deltas instead of absolute timestamps so that
        pauses don't artificially lower the measured FPS.
        """
        now = time.time()
        with self.output_fps_lock:
            if self._last_frame_time is not None:
                delta = now - self._last_frame_time
                self.output_frame_deltas.append(delta)

            self._last_frame_time = now

        self._calculate_output_fps()

    def _calculate_output_fps(self):
        """Calculate FPS from the average inter-frame delta."""
        with self.output_fps_lock:
            if len(self.output_frame_deltas) >= OUTPUT_FPS_MIN_SAMPLES:
                avg_delta = sum(self.output_frame_deltas) / len(
                    self.output_frame_deltas
                )
                if avg_delta > 0:
                    estimated_fps = 1.0 / avg_delta
                    # Clamp to reasonable bounds
                    estimated_fps = max(MIN_FPS, min(MAX_FPS, estimated_fps))
                    self.current_output_fps = estimated_fps

    def get_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS.

        Returns the FPS based on how fast frames are produced into the output queue,
        adjusted for queue fill level to prevent buildup.
        """
        with self.output_fps_lock:
            output_fps = self.current_output_fps
        return min(MAX_FPS, output_fps)

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        return True
