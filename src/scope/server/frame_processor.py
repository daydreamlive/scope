import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from aiortc.mediastreams import VideoFrame

from .kafka_publisher import publish_event
from .modulation import ModulationEngine
from .parameter_scheduler import ParameterScheduler
from .pipeline_manager import PipelineManager
from .pipeline_processor import PipelineProcessor

if TYPE_CHECKING:
    from av import AudioFrame

    from scope.core.inputs import InputSource
    from scope.core.outputs import OutputSink

    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


# FPS calculation constants
DEFAULT_FPS = 30.0  # Default FPS

# Heartbeat interval for stream stats logging and Kafka events
HEARTBEAT_INTERVAL_SECONDS = 10.0


class FrameProcessor:
    """Processes video frames through pipelines or cloud relay.

    Supports two modes:
    1. Local mode: Frames processed through local GPU pipelines
    2. Cloud mode: Frames sent to cloud for processing

    Output sink integration (NDI, Spout, etc.) works in both modes.
    """

    def __init__(
        self,
        pipeline_manager: "PipelineManager | None" = None,
        max_parameter_queue_size: int = 8,
        initial_parameters: dict = None,
        notification_callback: callable = None,
        cloud_manager: "CloudConnectionManager | None" = None,
        session_id: str | None = None,  # Session ID for event tracking
        user_id: str | None = None,  # User ID for event tracking
        connection_id: str | None = None,  # Connection ID for event correlation
        connection_info: dict
        | None = None,  # Connection metadata (gpu_type, region, etc.)
        tempo_sync: Any | None = None,
    ):
        self.pipeline_manager = pipeline_manager
        self.cloud_manager = cloud_manager
        self.tempo_sync = tempo_sync

        # Parameter scheduler for beat-synced parameter changes
        self.parameter_scheduler: ParameterScheduler | None = (
            ParameterScheduler(
                tempo_sync, self.update_parameters, notification_callback
            )
            if tempo_sync is not None
            else None
        )

        # Modulation engine for continuous beat-synced parameter oscillation
        self.modulation_engine: ModulationEngine | None = (
            ModulationEngine() if tempo_sync is not None else None
        )

        # Session ID for Kafka event tracking
        self.session_id = session_id or str(uuid.uuid4())
        # User ID for Kafka event tracking
        self.user_id = user_id
        # Connection ID from fal.ai WebSocket for event correlation
        self.connection_id = connection_id
        # Connection metadata (gpu_type, region, etc.) for Kafka events
        self.connection_info = connection_info

        # Current parameters
        self.parameters = initial_parameters or {}

        self.running = False

        # Callback to notify when frame processor stops
        self.notification_callback = notification_callback

        self.paused = False

        # Per-thread pinned buffers for H→D upload (local mode). Avoids sharing one
        # buffer across WebRTC/NDI/Spout threads (race) without a global lock that
        # serializes every frame.
        self._thread_pin_local = threading.local()

        # Cloud mode: send frames to cloud instead of local processing
        self._cloud_mode = cloud_manager is not None
        self._cloud_output_queue: queue.Queue = queue.Queue(maxsize=2)
        self._cloud_audio_queue: queue.Queue = queue.Queue(maxsize=50)
        self._frames_to_cloud = 0
        self._frames_from_cloud = 0
        # Track index for the primary browser source in cloud mode.
        # Defaults to 0; overridden when the graph has non-WebRTC sources
        # that shift the WebRTC source track positions.
        self._primary_cloud_track_index: int = 0

        # Output sinks keyed by type
        self.output_sinks: dict[str, dict] = {}

        self.input_source: InputSource | None = None
        self.input_source_enabled = False
        self.input_source_type = ""
        self.input_source_thread = None

        # Input mode: video waits for frames, text generates immediately
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Pipeline processors and IDs (populated by _setup_graph)
        self.pipeline_processors: list[PipelineProcessor] = []
        self.pipeline_ids: list[str] = []

        # Graph support: processors indexed by node_id for per-node routing
        self._processors_by_node_id: dict[str, PipelineProcessor] = {}
        # Graph source queues for fan-out from source nodes
        self._graph_source_queues: list[queue.Queue] = []
        # The processor whose output we read in graph mode
        self._sink_processor: PipelineProcessor | None = None

        # Multi-source/sink support: per-node queue routing
        self._source_queues_by_node: dict[str, list[queue.Queue]] = {}
        self._sink_queues_by_node: dict[str, queue.Queue] = {}
        # NDI/Spout/Syphon: duplicate fan-out queue (see graph_executor.GraphRun)
        self._sink_hardware_queues_by_node: dict[str, queue.Queue] = {}
        # Per-sink-node feeder processors for per-sink FPS
        self._sink_processors_by_node: dict[str, PipelineProcessor] = {}
        # Per-source-node input sources: node_id -> {source, thread, type}
        self._input_sources_by_node: dict[str, dict] = {}
        # Per-sink-node output sinks: node_id -> {sink, thread, type, name}
        self._output_sinks_by_node: dict[str, dict] = {}
        # Per-record-node queues and recording managers
        self._record_queues_by_node: dict[str, queue.Queue] = {}
        self._recording_managers_by_node: dict[str, dict] = {}

        # Frame counting for debug logging
        self._frames_in = 0
        self._frames_out = 0
        self._last_stats_time = time.time()
        self._last_heartbeat_time = time.time()
        self._playback_ready_emitted = False
        self._stream_start_time: float | None = None

        # Store pipeline_ids from initial_parameters if provided
        pipeline_ids = (initial_parameters or {}).get("pipeline_ids")
        if pipeline_ids is not None:
            self.pipeline_ids = pipeline_ids

    def start(self):
        if self.running:
            return

        self.running = True

        # Process output sink settings from initial parameters
        if "output_sinks" in self.parameters:
            sinks_config = self.parameters.pop("output_sinks")
            self._update_output_sinks_from_config(sinks_config)

        # Process generic input source settings.
        # Skip if a graph config with per-node sources is present — those are
        # handled by _setup_multi_input_sources and the legacy global mechanism
        # would broadcast frames to ALL source queues.
        if "input_source" in self.parameters:
            if self._graph_has_per_node_sources():
                self.parameters.pop("input_source")
                logger.info("Skipping legacy input_source — graph has per-node sources")
            else:
                input_source_config = self.parameters.pop("input_source")
                self._update_input_source(input_source_config)

        # Reset frame counters on start
        self._frames_in = 0
        self._frames_out = 0
        self._frames_to_cloud = 0
        self._frames_from_cloud = 0
        self._last_heartbeat_time = time.time()
        self._playback_ready_emitted = False
        self._stream_start_time = time.monotonic()
        self._last_stats_time = time.time()

        if self._cloud_mode:
            # Cloud mode: frames go to cloud instead of local pipelines
            logger.info("[FRAME-PROCESSOR] Starting in CLOUD mode (cloud)")

            # Register callbacks to receive frames from cloud
            if self.cloud_manager:
                self.cloud_manager.add_frame_callback(self._on_frame_from_cloud)
                self.cloud_manager.add_audio_callback(self._on_audio_from_cloud)

            # Set up per-node input sources (Syphon/NDI/Spout) and record
            # queues in cloud mode.  Also compute the cloud input track index
            # for the primary browser source (first WebRTC source node).
            graph_data = self.parameters.get("graph")
            if graph_data and isinstance(graph_data, dict):
                from .graph_schema import GraphConfig

                graph = GraphConfig(**graph_data)

                # Find the track index for the first WebRTC source so that
                # put() sends browser frames to the correct cloud input track.
                if self.cloud_manager:
                    for node in graph.nodes:
                        if node.type == "source":
                            sm = getattr(node, "source_mode", "video") or "video"
                            if sm not in ("spout", "ndi", "syphon"):
                                idx = self.cloud_manager.get_source_track_index(node.id)
                                if idx is not None:
                                    self._primary_cloud_track_index = idx
                                break

                if self._graph_has_per_node_sources():
                    self._setup_multi_input_sources(graph)

                # Set up record queues so cloud record frames can be
                # received locally for recording.
                for rec_id in graph.get_record_node_ids():
                    self._record_queues_by_node[rec_id] = queue.Queue(maxsize=30)
                if self._record_queues_by_node:
                    logger.info(
                        f"[FRAME-PROCESSOR] Cloud mode: created record queues "
                        f"for {list(self._record_queues_by_node.keys())}"
                    )

            logger.info("[FRAME-PROCESSOR] Started in cloud mode")

            # Publish stream_started event for relay mode
            publish_event(
                event_type="stream_started",
                session_id=self.session_id,
                connection_id=self.connection_id,
                user_id=self.user_id,
                metadata={"mode": "relay"},
                connection_info=self.connection_info,
            )
            return

        # Local mode: setup pipeline graph
        if not self.pipeline_ids:
            error_msg = "No pipeline IDs provided, cannot start"
            logger.error(error_msg)
            self.running = False
            # Publish error for startup failure
            publish_event(
                event_type="error",
                session_id=self.session_id,
                connection_id=self.connection_id,
                user_id=self.user_id,
                error={
                    "error_type": "stream_startup_failed",
                    "message": error_msg,
                    "exception_type": "ConfigurationError",
                    "recoverable": False,
                },
                metadata={"mode": "local"},
                connection_info=self.connection_info,
            )
            return

        try:
            self._setup_pipelines_sync()
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            self.running = False
            # Publish error for pipeline setup failure
            publish_event(
                event_type="error",
                session_id=self.session_id,
                connection_id=self.connection_id,
                pipeline_ids=self.pipeline_ids,
                user_id=self.user_id,
                error={
                    "error_type": "stream_startup_failed",
                    "message": str(e),
                    "exception_type": type(e).__name__,
                    "recoverable": False,
                },
                metadata={"mode": "local"},
                connection_info=self.connection_info,
            )
            return

        logger.info(
            f"[FRAME-PROCESSOR] Started with {len(self.pipeline_ids)} pipeline(s): {self.pipeline_ids}"
        )

        # Publish stream_started event for local mode
        publish_event(
            event_type="stream_started",
            session_id=self.session_id,
            connection_id=self.connection_id,
            pipeline_ids=self.pipeline_ids,
            user_id=self.user_id,
            metadata={"mode": "local"},
            connection_info=self.connection_info,
        )

    def stop(self, error_message: str = None):
        if not self.running:
            return

        self.running = False

        # Cancel any pending scheduled parameter changes
        if self.parameter_scheduler is not None:
            self.parameter_scheduler.cancel_pending()

        # Stop all pipeline processors
        for processor in self.pipeline_processors:
            processor.stop()

        # Clear pipeline processors
        self.pipeline_processors.clear()

        # Clear audio queue on the sink processor
        if self._sink_processor is not None:
            while not self._sink_processor.audio_output_queue.empty():
                try:
                    self._sink_processor.audio_output_queue.get_nowait()
                except queue.Empty:
                    break

        # Clean up all output sinks
        for sink_type, entry in list(self.output_sinks.items()):
            q = entry["queue"]
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            q.put_nowait(None)

            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        f"Output sink thread '{sink_type}' did not stop within 2s"
                    )
            try:
                entry["sink"].close()
            except Exception as e:
                logger.error(f"Error closing output sink '{sink_type}': {e}")
        self.output_sinks.clear()

        # Clean up generic input source
        self.input_source_enabled = False
        if self.input_source is not None:
            try:
                self.input_source.close()
            except Exception as e:
                logger.error(f"Error closing input source: {e}")
            self.input_source = None

        # Clean up per-node input sources (multi-source graph mode)
        # Join threads first to avoid closing the source while the thread is
        # still inside receive_frame() (causes segfault with PyAV/FFmpeg).
        for node_id, entry in list(self._input_sources_by_node.items()):
            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.warning(
                        f"Input source thread for node '{node_id}' "
                        f"did not stop within 3s"
                    )
            try:
                entry["source"].close()
            except Exception as e:
                logger.error(f"Error closing input source for node {node_id}: {e}")
        self._input_sources_by_node.clear()

        # Clean up per-node output sinks (multi-sink graph mode)
        for node_id, entry in list(self._output_sinks_by_node.items()):
            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning(
                        f"Output sink thread for node '{node_id}' "
                        f"did not stop within 2s"
                    )
            try:
                entry["sink"].close()
            except Exception as e:
                logger.error(f"Error closing output sink for node {node_id}: {e}")
        self._output_sinks_by_node.clear()

        # Clean up per-node recording managers
        for node_id, entry in list(self._recording_managers_by_node.items()):
            try:
                entry["track"].stop()
            except Exception as e:
                logger.error(f"Error stopping record track for node {node_id}: {e}")
        self._recording_managers_by_node.clear()
        self._record_queues_by_node.clear()

        # Clean up cloud callbacks in cloud mode
        if self._cloud_mode and self.cloud_manager:
            self.cloud_manager.remove_frame_callback(self._on_frame_from_cloud)
            self.cloud_manager.remove_audio_callback(self._on_audio_from_cloud)

        # Log final frame stats
        if self._cloud_mode:
            logger.info(
                f"[FRAME-PROCESSOR] Stopped (cloud mode). "
                f"Frames: in={self._frames_in}, to_cloud={self._frames_to_cloud}, "
                f"from_cloud={self._frames_from_cloud}, out={self._frames_out}"
            )
        else:
            logger.info(
                f"[FRAME-PROCESSOR] Stopped. Total frames: in={self._frames_in}, out={self._frames_out}"
            )

        # Notify callback that frame processor has stopped
        if self.notification_callback:
            try:
                message = {"type": "stream_stopped"}
                if error_message:
                    message["error_message"] = error_message
                self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error in frame processor stop callback: {e}")
        # Publish Kafka events for stream stop
        if error_message:
            # Publish error event for stream failure
            publish_event(
                event_type="error",
                session_id=self.session_id,
                connection_id=self.connection_id,
                pipeline_ids=self.pipeline_ids if self.pipeline_ids else None,
                user_id=self.user_id,
                error={
                    "error_type": "stream_failed",
                    "message": error_message,
                    "exception_type": "StreamError",
                    "recoverable": False,
                },
                metadata={
                    "mode": "cloud" if self._cloud_mode else "local",
                    "frames_in": self._frames_in,
                    "frames_out": self._frames_out,
                },
                connection_info=self.connection_info,
            )

        # Publish stream_stopped event
        publish_event(
            event_type="stream_stopped",
            session_id=self.session_id,
            connection_id=self.connection_id,
            pipeline_ids=self.pipeline_ids if self.pipeline_ids else None,
            user_id=self.user_id,
            metadata={
                "mode": "cloud" if self._cloud_mode else "local",
                "frames_in": self._frames_in,
                "frames_out": self._frames_out,
            },
            connection_info=self.connection_info,
        )

    def _thread_local_pinned_buffer(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Pinned host tensor for the current thread and frame shape."""
        if not hasattr(self._thread_pin_local, "buffers"):
            self._thread_pin_local.buffers = {}
        buf_map: dict[tuple[int, ...], torch.Tensor] = self._thread_pin_local.buffers
        if shape not in buf_map:
            buf_map[shape] = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        return buf_map[shape]

    def _frame_array_to_gpu(self, frame_array) -> torch.Tensor:
        """Convert a numpy frame array to a GPU tensor using pinned memory.

        Uses a **per-thread** pinned buffer so WebRTC, NDI, and Spout threads can
        upload concurrently (no shared buffer, no global lock). Within one thread,
        ``non_blocking=False`` ensures the H→D copy finishes before the next
        ``copy_`` overwrites the pinned buffer.
        """
        shape = tuple(frame_array.shape)
        pinned_buffer = self._thread_local_pinned_buffer(shape)
        pinned_buffer.copy_(torch.as_tensor(frame_array, dtype=torch.uint8))
        return pinned_buffer.cuda(non_blocking=False)

    def _frame_array_to_tensor(self, frame_array) -> torch.Tensor:
        """Convert a numpy frame array to a batched tensor (CPU or GPU)."""
        if torch.cuda.is_available():
            t = self._frame_array_to_gpu(frame_array)
        else:
            t = torch.as_tensor(frame_array, dtype=torch.uint8)
        return t.unsqueeze(0)

    def _maybe_emit_frame_heartbeat(self) -> None:
        """Log stats periodically when frames flow (shared by put() paths)."""
        now = time.time()
        if now - self._last_heartbeat_time >= HEARTBEAT_INTERVAL_SECONDS:
            self._log_frame_stats()
            self._last_heartbeat_time = now

    def put(self, frame: VideoFrame) -> bool:
        if not self.running:
            return False

        if self._cloud_mode:
            self._frames_in += 1
            self._maybe_emit_frame_heartbeat()
            # Cloud mode: send frame to cloud (only in video mode)
            # In text mode, cloud generates video from prompts only - no input frames
            if not self._video_mode:
                return True  # Silently ignore frames in text mode
            if self.cloud_manager:
                frame_array = frame.to_ndarray(format="rgb24")
                if self._primary_cloud_track_index != 0:
                    sent = self.cloud_manager.send_frame_to_track(
                        frame_array, self._primary_cloud_track_index
                    )
                else:
                    sent = self.cloud_manager.send_frame(frame_array)
                if sent:
                    self._frames_to_cloud += 1
                    return True
                else:
                    logger.debug("[FRAME-PROCESSOR] Failed to send frame to cloud")
                    return False
            return False

        # Local mode: put into graph source queues
        if not self._graph_source_queues:
            self._frames_in += 1
            self._maybe_emit_frame_heartbeat()
            return False

        # Multi-source graphs: legacy put() must not fan out to every source
        # queue (that mixes one input into all pipelines). Single-source graphs
        # (including one source with fan-out edges) have exactly one entry in
        # _source_queues_by_node — route the same as put_to_source.
        if len(self._source_queues_by_node) > 1:
            logger.warning(
                "Ignoring legacy put(frame): graph has multiple source nodes; "
                "use put_to_source() per source (WebRTC multi-source)"
            )
            return False

        self._frames_in += 1
        self._maybe_emit_frame_heartbeat()

        if len(self._source_queues_by_node) == 1:
            only_id = next(iter(self._source_queues_by_node))
            return self.put_to_source(frame, only_id, count_frame=False)

        return False

    def put_to_source(
        self,
        frame: VideoFrame,
        source_node_id: str,
        *,
        count_frame: bool = True,
    ) -> bool:
        """Route a frame to a specific source node's queues only (multi-source)."""
        if not self.running:
            return False

        queues = self._source_queues_by_node.get(source_node_id)
        if not queues:
            return False

        if count_frame:
            self._frames_in += 1

        frame_tensor = self._frame_array_to_tensor(frame.to_ndarray(format="rgb24"))

        for sq in queues:
            try:
                sq.put_nowait(frame_tensor)
            except queue.Full:
                logger.debug(
                    "Source node %s queue full, dropping frame", source_node_id
                )

        return True

    def get_from_sink(self, sink_node_id: str) -> torch.Tensor | None:
        """Read a frame from a specific sink node's output queue (multi-sink)."""
        if not self.running:
            return None

        sink_q = self._sink_queues_by_node.get(sink_node_id)
        if sink_q is None:
            return None

        try:
            frame = sink_q.get_nowait()
            frame = frame.squeeze(0)
            if frame.is_cuda:
                frame = frame.cpu()
            self._frames_out += 1
            return frame
        except queue.Empty:
            return None

    def get_from_record(self, record_node_id: str) -> torch.Tensor | None:
        """Read a frame from a specific record node's output queue."""
        if not self.running:
            return None

        rec_q = self._record_queues_by_node.get(record_node_id)
        if rec_q is None:
            return None

        try:
            frame = rec_q.get_nowait()
            frame = frame.squeeze(0)
            if frame.is_cuda:
                frame = frame.cpu()
            return frame
        except queue.Empty:
            return None

    def put_to_record(self, record_node_id: str, frame) -> None:
        """Put a VideoFrame into a record node's queue (cloud mode).

        Called by cloud output callbacks to populate record queues with
        frames received from the cloud pipeline.
        """
        rec_q = self._record_queues_by_node.get(record_node_id)
        if rec_q is None:
            return
        try:
            frame_np = frame.to_ndarray(format="rgb24")
            frame_tensor = torch.as_tensor(frame_np, dtype=torch.uint8).unsqueeze(0)
            try:
                rec_q.put_nowait(frame_tensor)
            except queue.Full:
                try:
                    rec_q.get_nowait()
                    rec_q.put_nowait(frame_tensor)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"Error putting frame to record node {record_node_id}: {e}")

    def get_sink_node_ids(self) -> list[str]:
        """Return the list of sink node IDs available for reading."""
        return list(self._sink_queues_by_node.keys())

    def get_unhandled_sink_node_ids(self) -> list[str]:
        """Return sink node IDs that don't have their own output sink thread.

        These sinks need external draining (e.g. by the headless consumer)
        to prevent their queues from filling up and stalling the pipeline.
        """
        return [
            sid
            for sid in self._sink_queues_by_node
            if sid not in self._output_sinks_by_node
        ]

    def get_record_node_ids(self) -> list[str]:
        """Return the list of record node IDs in the graph."""
        return list(self._record_queues_by_node.keys())

    async def start_node_recording(self, node_id: str) -> bool:
        """Start recording for a specific record node."""
        rec_q = self._record_queues_by_node.get(node_id)
        if rec_q is None:
            logger.error(f"No record queue for node {node_id}")
            return False

        if node_id in self._recording_managers_by_node:
            entry = self._recording_managers_by_node[node_id]
            if entry["manager"].is_recording_started:
                logger.info(f"Record node {node_id} already recording")
                return True

        from .recording import RecordingManager
        from .tracks import QueueVideoTrack

        track = QueueVideoTrack(rec_q, fps=self.get_fps())
        manager = RecordingManager(video_track=track)
        # No MediaRelay: this track is only consumed by RecordingManager.
        # relay.subscribe() is for sharing one source with WebRTC + recorder;
        # for a queue-backed track it can prevent MediaRecorder from receiving frames.

        self._recording_managers_by_node[node_id] = {
            "manager": manager,
            "track": track,
        }

        await manager.start_recording()
        logger.info(f"Started recording for record node {node_id}")
        return True

    async def stop_node_recording(self, node_id: str) -> bool:
        """Stop recording for a specific record node."""
        entry = self._recording_managers_by_node.get(node_id)
        if entry is None:
            return False
        await entry["manager"].stop_recording()
        logger.info(f"Stopped recording for record node {node_id}")
        return True

    async def download_node_recording(self, node_id: str) -> str | None:
        """Finalize and return the recording file path for a record node."""
        entry = self._recording_managers_by_node.get(node_id)
        if entry is None:
            return None
        path = await entry["manager"].finalize_and_get_recording(restart_after=False)
        self._recording_managers_by_node.pop(node_id, None)
        return path

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        # Get frame based on mode
        frame: torch.Tensor | None = None

        if self._cloud_mode:
            # Cloud mode: get frame from cloud output queue
            try:
                frame_np = self._cloud_output_queue.get_nowait()
                frame = torch.from_numpy(frame_np)
            except queue.Empty:
                return None
        else:
            # Local mode: get from pipeline processor
            if not self.pipeline_processors:
                return None

            if self._sink_processor is None or not self._sink_processor.output_queue:
                return None

            try:
                frame = self._sink_processor.output_queue.get_nowait()
                # Frame is stored as [1, H, W, C], convert to [H, W, C] for output
                # Move to CPU here for WebRTC streaming (frames stay on GPU between pipeline processors)
                frame = frame.squeeze(0)
                if frame.is_cuda:
                    frame = frame.cpu()
            except queue.Empty:
                return None

        self._on_frame_output(frame)
        return frame

    def get_audio(self) -> tuple[torch.Tensor | None, int | None]:
        """Get the next audio chunk and its sample rate.

        In local mode, reads from the sink processor's audio output queue.
        In cloud mode, reads from the cloud audio queue (populated by
        _on_audio_from_cloud).

        Returns:
            Tuple of (audio_tensor, sample_rate) or (None, None) if no audio available.
            audio_tensor shape: (channels, samples) - typically (2, N) for stereo
        """
        if not self.running:
            return None, None

        if self._cloud_mode:
            try:
                audio, sample_rate = self._cloud_audio_queue.get_nowait()
                return audio, sample_rate
            except queue.Empty:
                return None, None

        if self._sink_processor is None:
            return None, None

        try:
            audio, sample_rate = self._sink_processor.audio_output_queue.get_nowait()
            # Pass through flush sentinels (audio=None, sample_rate=-1)
            return audio, sample_rate
        except queue.Empty:
            return None, None

    def _on_frame_from_cloud(self, frame: "VideoFrame") -> None:
        """Callback when a processed frame is received from cloud (cloud mode)."""
        self._frames_from_cloud += 1

        try:
            # Convert to numpy and queue for output
            frame_np = frame.to_ndarray(format="rgb24")
            try:
                self._cloud_output_queue.put_nowait(frame_np)
            except queue.Full:
                # Drop oldest frame to make room
                try:
                    self._cloud_output_queue.get_nowait()
                    self._cloud_output_queue.put_nowait(frame_np)
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"[FRAME-PROCESSOR] Error processing frame from cloud: {e}")

    def _on_audio_from_cloud(self, frame: "AudioFrame") -> None:
        """Callback when an audio frame is received from cloud (cloud mode).

        Converts the AudioFrame to a torch tensor and queues it for
        AudioProcessingTrack to consume via get_audio().

        Packed formats (s16) store interleaved channels in a single plane,
        so to_ndarray() returns (1, samples*channels).  We de-interleave
        into (channels, samples) so AudioProcessingTrack sees the correct
        channel count and doesn't erroneously duplicate data.
        """
        try:
            n_channels = len(frame.layout.channels)
            audio_np = frame.to_ndarray()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            # Packed formats (e.g. s16) have 1 plane with interleaved channels:
            # [L0, R0, L1, R1, ...].  De-interleave into (channels, samples).
            if audio_np.shape[0] == 1 and n_channels > 1:
                flat = audio_np.ravel()
                audio_np = flat.reshape(-1, n_channels).T

            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))

            # Normalise int16 range to [-1, 1] float if needed
            if frame.format.name in ("s16", "s16p"):
                audio_tensor = audio_tensor / 32768.0

            try:
                self._cloud_audio_queue.put_nowait((audio_tensor, frame.sample_rate))
            except queue.Full:
                # Drop oldest to keep latency low
                try:
                    self._cloud_audio_queue.get_nowait()
                    self._cloud_audio_queue.put_nowait(
                        (audio_tensor, frame.sample_rate)
                    )
                except queue.Empty:
                    pass
        except Exception as e:
            logger.error(f"[FRAME-PROCESSOR] Error processing audio from cloud: {e}")

    def get_fps(self) -> float:
        """Get the playback FPS for the video track.

        Delegates to the last pipeline processor which returns native_fps
        (e.g. 24fps) when the pipeline reports it, or the measured production
        rate otherwise.
        """
        if not self.pipeline_processors:
            return DEFAULT_FPS

        if self._sink_processor is None:
            return DEFAULT_FPS
        return self._sink_processor.get_fps()

    def get_fps_for_sink(self, sink_node_id: str) -> float:
        """Get FPS for a specific sink node from its feeder processor."""
        proc = self._sink_processors_by_node.get(sink_node_id)
        if proc is not None:
            return proc.get_fps()
        return self.get_fps()

    def notify_primary_frame_output(self, frame: torch.Tensor) -> None:
        """Handle side effects for frames from the primary output track.

        Called by the primary track's recv() when using get_from_sink()
        instead of get(). Emits the playback_ready event on first frame
        and fans out to output sinks (NDI/Spout).
        """
        self._on_frame_output(frame)

    def _on_frame_output(self, frame: torch.Tensor) -> None:
        """Common post-output logic: increment counter, emit playback_ready, fan out to sinks."""
        self._frames_out += 1

        if not self._playback_ready_emitted:
            self._playback_ready_emitted = True
            time_to_first_frame_ms = (
                int((time.monotonic() - self._stream_start_time) * 1000)
                if self._stream_start_time is not None
                else None
            )
            publish_event(
                event_type="playback_ready",
                session_id=self.session_id,
                connection_id=self.connection_id,
                pipeline_ids=self.pipeline_ids if self.pipeline_ids else None,
                user_id=self.user_id,
                metadata={
                    "mode": "cloud" if self._cloud_mode else "local",
                    "ttff_ms": time_to_first_frame_ms,
                },
                connection_info=self.connection_info,
            )
            logger.info(
                f"[FRAME-PROCESSOR] First frame produced, playback ready "
                f"(session={self.session_id}, mode={'cloud' if self._cloud_mode else 'local'}, "
                f"ttff={time_to_first_frame_ms}ms)"
            )

        if self.output_sinks:
            try:
                frame_np = frame.numpy()
                for _sink_type, entry in self.output_sinks.items():
                    try:
                        entry["queue"].put_nowait(frame_np)
                    except queue.Full:
                        pass
            except Exception as e:
                logger.error(f"Error enqueueing output sink frame: {e}")

    def _log_frame_stats(self):
        """Log frame processing statistics and emit heartbeat event."""
        now = time.time()
        elapsed = now - self._last_stats_time

        if elapsed > 0:
            fps_in = self._frames_in / elapsed if self._frames_in > 0 else 0
            fps_out = self._frames_out / elapsed if self._frames_out > 0 else 0
            pipeline_fps = self.get_fps() if not self._cloud_mode else None

            if self._cloud_mode:
                logger.info(
                    f"[FRAME-PROCESSOR] RELAY MODE | "
                    f"Frames: in={self._frames_in}, to_cloud={self._frames_to_cloud}, "
                    f"from_cloud={self._frames_from_cloud}, out={self._frames_out} | "
                    f"Rate: {fps_in:.1f} fps in, {fps_out:.1f} fps out"
                )
            else:
                logger.info(
                    f"[FRAME-PROCESSOR] Frames: in={self._frames_in}, out={self._frames_out} | "
                    f"Rate: {fps_in:.1f} fps in, {fps_out:.1f} fps out | "
                    f"Pipeline FPS: {pipeline_fps:.1f}"
                )

            # Emit stream_heartbeat Kafka event
            heartbeat_metadata = {
                "mode": "cloud" if self._cloud_mode else "local",
                "frames_in": self._frames_in,
                "frames_out": self._frames_out,
                "fps_in": round(fps_in, 1),
                "fps_out": round(fps_out, 1),
                "elapsed_ms": int(elapsed * 1000),
                "since_last_heartbeat_ms": int(
                    (now - self._last_heartbeat_time) * 1000
                ),
            }
            if self._cloud_mode:
                heartbeat_metadata["frames_to_cloud"] = self._frames_to_cloud
                heartbeat_metadata["frames_from_cloud"] = self._frames_from_cloud
            else:
                heartbeat_metadata["pipeline_fps"] = (
                    round(pipeline_fps, 1) if pipeline_fps else None
                )

            publish_event(
                event_type="stream_heartbeat",
                session_id=self.session_id,
                connection_id=self.connection_id,
                pipeline_ids=self.pipeline_ids if self.pipeline_ids else None,
                user_id=self.user_id,
                metadata=heartbeat_metadata,
                connection_info=self.connection_info,
            )

    def get_frame_stats(self) -> dict:
        """Get current frame processing statistics."""
        now = time.time()
        elapsed = now - self._last_stats_time

        stats = {
            "frames_in": self._frames_in,
            "frames_out": self._frames_out,
            "elapsed_seconds": elapsed,
            "fps_in": self._frames_in / elapsed if elapsed > 0 else 0,
            "fps_out": self._frames_out / elapsed if elapsed > 0 else 0,
            "pipeline_fps": self.get_fps(),
            "output_sinks": {
                k: {"name": v["name"]} for k, v in self.output_sinks.items()
            },
            "input_source_enabled": self.input_source_enabled,
            "input_source_type": self.input_source_type,
            "relay_mode": self._cloud_mode,
        }

        if self._cloud_mode:
            stats["frames_to_cloud"] = self._frames_to_cloud
            stats["frames_from_cloud"] = self._frames_from_cloud

        return stats

    def _get_pipeline_dimensions(self) -> tuple[int, int]:
        """Get current pipeline dimensions from pipeline manager."""
        try:
            status_info = self.pipeline_manager.get_status_info()
            load_params = status_info.get("load_params") or {}
            width = load_params.get("width", 512)
            height = load_params.get("height", 512)
            return width, height
        except Exception as e:
            logger.warning(f"Could not get pipeline dimensions: {e}")
            return 512, 512

    def schedule_quantized_update(self, params: dict):
        """Schedule params to be applied at the next beat boundary."""
        if self.parameter_scheduler is not None:
            self.parameter_scheduler.schedule(params)
        else:
            self.update_parameters(params)

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        # Always strip tempo-control keys so they never leak into pipelines,
        # even when the corresponding helper (scheduler/engine/tempo_sync) is absent.

        if "quantize_mode" in parameters:
            mode = parameters.pop("quantize_mode")
            if self.parameter_scheduler is not None:
                self.parameter_scheduler.quantize_mode = mode

        if "lookahead_ms" in parameters:
            ms = parameters.pop("lookahead_ms")
            if self.parameter_scheduler is not None:
                self.parameter_scheduler.lookahead_ms = ms

        # Handle generic output sinks config
        if "output_sinks" in parameters:
            sinks_config = parameters.pop("output_sinks")
            self._update_output_sinks_from_config(sinks_config)

        # Handle generic input source settings.
        # Skip when per-node graph sources are active to avoid broadcasting
        # frames to all queues (the per-node threads handle routing).
        if "input_source" in parameters:
            if self._input_sources_by_node:
                parameters.pop("input_source")
            else:
                input_source_config = parameters.pop("input_source")
                self._update_input_source(input_source_config)

        if "modulations" in parameters:
            raw = parameters.pop("modulations")
            if self.modulation_engine is not None:
                self.modulation_engine.update(raw)

        if "beat_cache_reset_rate" in parameters:
            rate = parameters.pop("beat_cache_reset_rate")
            for processor in self.pipeline_processors:
                processor.set_beat_cache_reset_rate(rate)

        # Strip client-forwarded beat state keys so they are never forwarded
        # as regular pipeline parameters (they are injected separately by
        # PipelineProcessor). Route to TempoSync when available.
        if self.tempo_sync is not None:
            parameters = self.tempo_sync.update_client_beat_state(parameters)
        else:
            from .tempo_sync import BEAT_STATE_KEYS

            parameters = {
                k: v for k, v in parameters.items() if k not in BEAT_STATE_KEYS
            }

        # Route to specific node or broadcast to all pipeline processors
        node_id = parameters.pop("node_id", None)
        if node_id:
            if node_id in self._processors_by_node_id:
                self._processors_by_node_id[node_id].update_parameters(parameters)
            else:
                logger.warning(
                    f"Unknown node_id '{node_id}', ignoring parameter update"
                )
        else:
            for processor in self.pipeline_processors:
                processor.update_parameters(parameters)

        # Update local parameters
        self.parameters = {**self.parameters, **parameters}

        return True

    def _update_output_sinks_from_config(self, config: dict):
        """Handle the generic output_sinks config dict.

        Config format: {"spout": {"enabled": True, "name": "ScopeOut"}, "ndi": {...}}
        """
        from scope.core.outputs import get_output_sink_classes

        sink_classes = get_output_sink_classes()
        for sink_type, sink_config in config.items():
            enabled = sink_config.get("enabled", False)
            name = sink_config.get("name", "")
            sink_cls = sink_classes.get(sink_type)
            if sink_cls is None:
                if enabled:
                    logger.warning(f"Output sink '{sink_type}' not available")
                continue
            self._update_output_sink(
                sink_type=sink_type,
                enabled=enabled,
                sink_name=name,
                sink_class=sink_cls,
            )

    def _update_output_sink(
        self,
        sink_type: str,
        enabled: bool,
        sink_name: str,
        sink_class: "type[OutputSink] | None" = None,
    ):
        """Create, update, or destroy a single output sink entry."""
        width, height = self._get_pipeline_dimensions()
        existing = self.output_sinks.get(sink_type)

        logger.info(
            f"Output sink config: type={sink_type}, enabled={enabled}, "
            f"name={sink_name}, size={width}x{height}"
        )

        if enabled and existing is None:
            # Create new sink
            if sink_class is None:
                from scope.core.outputs import get_output_sink_classes

                sink_class = get_output_sink_classes().get(sink_type)
            if sink_class is None:
                logger.error(f"Unknown output sink type: {sink_type}")
                return
            try:
                sink = sink_class()
                if sink.create(sink_name, width, height):
                    q: queue.Queue = queue.Queue(maxsize=30)
                    t = threading.Thread(
                        target=self._output_sink_loop,
                        args=(sink_type,),
                        daemon=True,
                    )
                    self.output_sinks[sink_type] = {
                        "sink": sink,
                        "queue": q,
                        "thread": t,
                        "name": sink_name,
                    }
                    t.start()
                    logger.info(f"Output sink enabled: {sink_type} '{sink_name}'")
                else:
                    logger.error(f"Failed to create output sink: {sink_type}")
            except Exception as e:
                logger.error(f"Error creating output sink '{sink_type}': {e}")

        elif not enabled and existing is not None:
            # Destroy existing sink
            self._close_output_sink(sink_type)
            logger.info(f"Output sink disabled: {sink_type}")

        elif enabled and existing is not None:
            # Recreate if name or dimensions changed
            old_sink = existing["sink"]
            needs_recreate = sink_name != existing["name"]
            if hasattr(old_sink, "width") and hasattr(old_sink, "height"):
                if old_sink.width != width or old_sink.height != height:
                    needs_recreate = True

            if needs_recreate:
                self._close_output_sink(sink_type)
                self._update_output_sink(
                    sink_type=sink_type,
                    enabled=True,
                    sink_name=sink_name,
                    sink_class=sink_class,
                )

    def _close_output_sink(self, sink_type: str):
        """Stop and remove a single output sink."""
        entry = self.output_sinks.pop(sink_type, None)
        if entry is None:
            return

        q = entry["queue"]
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
        q.put_nowait(None)

        thread = entry.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    f"Output sink thread '{sink_type}' did not stop within 2s"
                )
        try:
            entry["sink"].close()
        except Exception as e:
            logger.error(f"Error closing output sink '{sink_type}': {e}")

    def _output_sink_loop(self, sink_type: str):
        """Background thread that sends frames for a single output sink."""
        logger.info(f"Output sink thread started: {sink_type}")
        frame_count = 0

        while self.running and sink_type in self.output_sinks:
            entry = self.output_sinks.get(sink_type)
            if entry is None:
                break
            try:
                try:
                    frame_np = entry["queue"].get(timeout=0.1)
                    if frame_np is None:
                        break
                except queue.Empty:
                    continue

                success = entry["sink"].send_frame(frame_np)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(
                        f"Output sink '{sink_type}' sent frame {frame_count}, "
                        f"shape={frame_np.shape}, success={success}"
                    )

            except Exception as e:
                logger.error(f"Error in output sink loop '{sink_type}': {e}")
                time.sleep(0.01)

        logger.info(f"Output sink thread stopped: {sink_type} ({frame_count} frames)")

    def _graph_has_per_node_sources(self) -> bool:
        """Check if the graph config has source nodes with non-WebRTC modes.

        When True, those sources will be handled by _setup_multi_input_sources
        and the legacy global input_source mechanism should be skipped.
        """
        graph_data = self.parameters.get("graph")
        if not graph_data or not isinstance(graph_data, dict):
            return False
        for node in graph_data.get("nodes", []):
            if node.get("type") == "source":
                sm = node.get("source_mode")
                if sm in ("spout", "ndi", "syphon", "video_file"):
                    return True
        return False

    def _update_input_source(self, config: dict):
        """Update generic input source configuration."""
        enabled = config.get("enabled", False)
        source_type = config.get("source_type", "")
        source_name = config.get("source_name", "")

        logger.info(
            f"Input source config: enabled={enabled}, "
            f"type={source_type}, name={source_name}"
        )

        if enabled and not self.input_source_enabled:
            self._create_and_connect_input_source(source_type, source_name)

        elif not enabled and self.input_source_enabled:
            self.input_source_enabled = False
            if self.input_source is not None:
                self.input_source.close()
                self.input_source = None
            logger.info("Input source disabled")

        elif enabled and (
            source_type != self.input_source_type or config.get("reconnect", False)
        ):
            self.input_source_enabled = False
            if self.input_source is not None:
                self.input_source.close()
                self.input_source = None
            self._create_and_connect_input_source(source_type, source_name)

    def _create_and_connect_input_source(self, source_type: str, source_name: str):
        """Create an input source instance and connect to the given source."""
        from scope.core.inputs import get_input_source_classes

        input_source_classes = get_input_source_classes()
        source_class = input_source_classes.get(source_type)

        if source_class is None:
            logger.error(
                f"Unknown input source type '{source_type}'. "
                f"Available: {list(input_source_classes.keys())}"
            )
            return

        if not source_class.is_available():
            logger.error(
                f"Input source '{source_type}' is not available on this platform"
            )
            return

        try:
            self.input_source = source_class()
            if self.input_source.connect(source_name):
                self.input_source_enabled = True
                self.input_source_type = source_type
                self.input_source_thread = threading.Thread(
                    target=self._input_source_receiver_loop, daemon=True
                )
                self.input_source_thread.start()
                logger.info(f"Input source enabled: {source_type} -> '{source_name}'")
            else:
                logger.error(
                    f"Failed to connect to input source: "
                    f"{source_type} -> '{source_name}'"
                )
                self.input_source.close()
                self.input_source = None
        except Exception as e:
            logger.error(f"Error creating input source '{source_type}': {e}")
            if self.input_source is not None:
                try:
                    self.input_source.close()
                except Exception:
                    pass
            self.input_source = None

    def _input_source_receiver_loop(self):
        """Background thread that receives frames from a generic input source.

        Receives frames as fast as the source provides them, without throttling
        based on pipeline FPS. Backpressure is handled by the downstream queues
        (put_nowait drops frames when full). This matches the behavior of the
        WebRTC camera input path and avoids a feedback loop where FPS-based
        throttling + receive latency causes a downward FPS spiral for sources
        with non-trivial receive overhead (NDI, Syphon).
        """
        logger.info(f"Input source thread started ({self.input_source_type})")

        frame_count = 0

        while (
            self.running and self.input_source_enabled and self.input_source is not None
        ):
            try:
                rgb_frame = self.input_source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    if self._cloud_mode:
                        if self._video_mode and self.cloud_manager:
                            if self.cloud_manager.send_frame(rgb_frame):
                                self._frames_to_cloud += 1
                    elif self._graph_source_queues:
                        frame_tensor = self._frame_array_to_tensor(rgb_frame)

                        for sq in self._graph_source_queues:
                            try:
                                sq.put_nowait(frame_tensor)
                            except queue.Full:
                                logger.debug(
                                    f"Graph source queue full, "
                                    f"dropping {self.input_source_type} frame"
                                )

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(
                            f"Input source ({self.input_source_type}) "
                            f"received {frame_count} frames"
                        )
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in input source loop: {e}")
                time.sleep(0.01)

        logger.info(
            f"Input source thread stopped ({self.input_source_type}) "
            f"after {frame_count} frames"
        )

    def _setup_pipelines_sync(self):
        """Create pipeline execution graph (synchronous).

        If a graph config is provided via initial parameters, uses build_graph()
        to create the execution graph. Otherwise, builds an implicit linear graph
        from pipeline_ids. Assumes all pipelines are already loaded by the
        pipeline manager.
        """
        from scope.core.pipelines.wan2_1.vace import VACEEnabledPipeline

        from .graph_schema import GraphConfig, build_linear_graph

        graph_data = self.parameters.get("graph")
        if graph_data is not None:
            api_graph = GraphConfig.model_validate(graph_data)

            # A graph without source nodes cannot receive video input.
            # Force text mode so pipeline processors don't wait forever
            # for frames that will never arrive (e.g. Workflow Builder
            # with no Source node connected to cloud).
            if not api_graph.get_source_node_ids():
                self.parameters["input_mode"] = "text"
                self._video_mode = False
        else:
            # Determine which pipelines should receive input as vace_input_frames
            vace_input_video_ids: set[str] = set()
            if self.parameters.get("vace_enabled") and self.parameters.get(
                "vace_use_input_video", True
            ):
                for pid in self.pipeline_ids:
                    pipeline = self.pipeline_manager.get_pipeline_by_id(pid)
                    if isinstance(pipeline, VACEEnabledPipeline):
                        vace_input_video_ids.add(pid)

            api_graph = build_linear_graph(
                self.pipeline_ids,
                vace_input_video_ids=vace_input_video_ids or None,
            )

        self._setup_graph(api_graph)

    def _setup_graph(self, graph):
        """Set up graph-based execution from a GraphConfig."""
        from .graph_executor import build_graph

        graph_run = build_graph(
            graph=graph,
            pipeline_manager=self.pipeline_manager,
            initial_parameters=self.parameters.copy(),
            session_id=self.session_id,
            user_id=self.user_id,
            connection_id=self.connection_id,
            connection_info=self.connection_info,
            tempo_sync=self.tempo_sync,
            modulation_engine=self.modulation_engine,
        )

        self._graph_source_queues = graph_run.source_queues
        self._sink_processor = graph_run.sink_processor
        self.pipeline_processors = graph_run.processors
        self.pipeline_ids = graph_run.pipeline_ids

        # Store per-node queue mappings for multi-source/sink/record
        self._source_queues_by_node = graph_run.source_queues_by_node
        self._sink_queues_by_node = graph_run.sink_queues_by_node
        self._sink_hardware_queues_by_node = graph_run.sink_hardware_queues_by_node
        self._sink_processors_by_node = graph_run.sink_processors_by_node
        self._record_queues_by_node = graph_run.record_queues_by_node

        # Index processors by node_id for per-node parameter routing
        for proc in self.pipeline_processors:
            self._processors_by_node_id[proc.node_id] = proc

        # Start all processors
        for processor in self.pipeline_processors:
            processor.start()

        # Set up per-source-node input sources for non-WebRTC sources
        self._setup_multi_input_sources(graph)

        # Set up per-sink-node output sinks for non-WebRTC sinks
        self._setup_multi_output_sinks(graph)

        logger.info(
            f"Created graph with {len(self.pipeline_processors)} processors, "
            f"sink={graph_run.sink_node_id}, "
            f"sources={list(self._source_queues_by_node.keys())}, "
            f"sinks={list(self._sink_queues_by_node.keys())}, "
            f"records={list(self._record_queues_by_node.keys())}"
        )

    def _setup_multi_input_sources(self, graph):
        """Set up per-source-node input sources for non-WebRTC graph sources.

        For source nodes with source_mode in (spout, ndi, syphon), creates a
        separate InputSource + receiver thread for each one.
        """
        from .graph_schema import GraphConfig

        if not isinstance(graph, GraphConfig):
            return

        for node in graph.nodes:
            if node.type != "source":
                continue
            source_mode = getattr(node, "source_mode", None)
            if source_mode not in ("spout", "ndi", "syphon", "video_file"):
                continue
            source_name = getattr(node, "source_name", "") or ""
            node_id = node.id

            # In cloud mode we don't need local source queues — frames are
            # forwarded to the cloud via send_frame_to_track. In local mode
            # the node must have queues registered by the graph executor.
            if not self._cloud_mode and node_id not in self._source_queues_by_node:
                continue

            from scope.core.inputs import get_input_source_classes

            input_source_classes = get_input_source_classes()
            source_class = input_source_classes.get(source_mode)
            if source_class is None or not source_class.is_available():
                logger.warning(
                    f"Input source '{source_mode}' not available for node {node_id}"
                )
                continue

            try:
                source = source_class()
                if source.connect(source_name):
                    thread = threading.Thread(
                        target=self._multi_input_source_loop,
                        args=(node_id, source_mode),
                        daemon=True,
                    )
                    self._input_sources_by_node[node_id] = {
                        "source": source,
                        "thread": thread,
                        "type": source_mode,
                    }
                    thread.start()
                    logger.info(
                        f"Multi-source: started {source_mode} for node {node_id}"
                    )
                else:
                    logger.error(
                        f"Failed to connect input source {source_mode} "
                        f"for node {node_id}"
                    )
                    source.close()
            except Exception as e:
                logger.error(
                    f"Error creating input source '{source_mode}' "
                    f"for node {node_id}: {e}"
                )

    def _multi_input_source_loop(self, node_id: str, source_type: str):
        """Background thread that receives frames for a specific source node.

        Receives as fast as the source provides frames, without throttling to
        pipeline output FPS. Throttling on measured get_fps() created a feedback
        loop: slower output → lower get_fps() → longer sleeps → starved inputs →
        even slower output. Backpressure is queue full + drop, same as
        :meth:`_input_source_receiver_loop`.
        """
        entry = self._input_sources_by_node.get(node_id)
        if entry is None:
            return

        source = entry["source"]
        frame_count = 0

        while self.running and node_id in self._input_sources_by_node:
            try:
                rgb_frame = source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    if self._cloud_mode:
                        # Cloud mode: forward frames to cloud manager
                        # Route to the correct input track for this source node
                        if self._video_mode and self.cloud_manager:
                            track_idx = self.cloud_manager.get_source_track_index(
                                node_id
                            )
                            if track_idx is not None:
                                sent = self.cloud_manager.send_frame_to_track(
                                    rgb_frame, track_idx
                                )
                            else:
                                sent = self.cloud_manager.send_frame(rgb_frame)
                            if sent:
                                self._frames_to_cloud += 1
                    else:
                        queues = self._source_queues_by_node.get(node_id)
                        if queues:
                            frame_tensor = self._frame_array_to_tensor(rgb_frame)
                            for sq in queues:
                                try:
                                    sq.put_nowait(frame_tensor)
                                except queue.Full:
                                    logger.debug(
                                        f"Multi-source node {node_id} queue full, "
                                        f"dropping {source_type} frame"
                                    )

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(
                            f"Multi-source ({source_type}) node {node_id}: "
                            f"{frame_count} frames"
                        )
                else:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in multi-source loop node {node_id}: {e}")
                time.sleep(0.01)

        logger.info(
            f"Multi-source thread stopped ({source_type}) node {node_id} "
            f"after {frame_count} frames"
        )

    def _setup_multi_output_sinks(self, graph):
        """Set up per-sink-node output sinks for non-WebRTC graph sinks.

        For sink nodes with sink_mode in (spout, ndi, syphon), creates a
        separate OutputSink + sender thread for each one.
        """
        from .graph_schema import GraphConfig

        if not isinstance(graph, GraphConfig):
            return

        for node in graph.nodes:
            if node.type != "sink":
                continue
            sink_mode = getattr(node, "sink_mode", None)
            if sink_mode not in ("spout", "ndi", "syphon"):
                continue
            sink_name = getattr(node, "sink_name", "") or ""
            node_id = node.id

            if node_id not in self._sink_queues_by_node:
                continue

            from scope.core.outputs import get_output_sink_classes

            sink_classes = get_output_sink_classes()
            sink_class = sink_classes.get(sink_mode)
            if sink_class is None:
                logger.warning(
                    f"Output sink '{sink_mode}' not available for node {node_id}"
                )
                continue

            try:
                width, height = self._get_pipeline_dimensions()
                sink = sink_class()
                if sink.create(sink_name, width, height):
                    thread = threading.Thread(
                        target=self._multi_output_sink_loop,
                        args=(node_id, sink_mode),
                        daemon=True,
                    )
                    self._output_sinks_by_node[node_id] = {
                        "sink": sink,
                        "thread": thread,
                        "type": sink_mode,
                        "name": sink_name,
                    }
                    thread.start()
                    logger.info(
                        f"Multi-sink: started {sink_mode} '{sink_name}' "
                        f"for node {node_id}"
                    )
                else:
                    logger.error(
                        f"Failed to create output sink {sink_mode} for node {node_id}"
                    )
                    sink.close()
            except Exception as e:
                logger.error(
                    f"Error creating output sink '{sink_mode}' for node {node_id}: {e}"
                )

    def _multi_output_sink_loop(self, node_id: str, sink_type: str):
        """Background thread that sends frames for a specific sink node."""
        entry = self._output_sinks_by_node.get(node_id)
        if entry is None:
            return

        sink = entry["sink"]
        sink_q = self._sink_hardware_queues_by_node.get(node_id)
        if sink_q is None:
            sink_q = self._sink_queues_by_node.get(node_id)
        if sink_q is None:
            logger.error(f"No sink queue for node {node_id}")
            return

        frame_count = 0
        logger.info(f"Multi-sink output thread started: {sink_type} node {node_id}")

        while self.running and node_id in self._output_sinks_by_node:
            try:
                try:
                    frame = sink_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Convert tensor to numpy for the output sink
                frame_squeezed = frame.squeeze(0)
                if frame_squeezed.is_cuda:
                    frame_squeezed = frame_squeezed.cpu()
                frame_np = frame_squeezed.numpy()

                sink.send_frame(frame_np)
                frame_count += 1

                if frame_count % 300 == 0:
                    logger.debug(
                        f"Multi-sink ({sink_type}) node {node_id}: "
                        f"{frame_count} frames sent"
                    )

            except Exception as e:
                logger.error(f"Error in multi-sink output loop node {node_id}: {e}")
                time.sleep(0.01)

        logger.info(
            f"Multi-sink output thread stopped ({sink_type}) node {node_id} "
            f"after {frame_count} frames"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
