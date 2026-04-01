import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from aiortc.mediastreams import VideoFrame

from .cloud_relay import CloudRelay
from .kafka_publisher import publish_event
from .modulation import ModulationEngine
from .parameter_scheduler import ParameterScheduler
from .pipeline_manager import PipelineManager
from .pipeline_processor import PipelineProcessor
from .sink_manager import SinkManager
from .source_manager import SourceManager

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


# FPS calculation constants
DEFAULT_FPS = 30.0  # Default FPS

# Heartbeat interval for stream stats logging and Kafka events
HEARTBEAT_INTERVAL_SECONDS = 10.0


class FrameProcessor:
    """Processes video frames through pipelines or cloud relay.

    Supports two modes (selected by presence of a CloudConnectionManager):
    1. Local mode: Frames processed through local GPU pipelines
    2. Cloud mode: Frames sent to cloud for processing

    Input/output routing (WebRTC, NDI, Spout, recording, etc.) is delegated
    to SourceManager and SinkManager.
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

        # Cloud relay (None in local mode)
        video_mode = (initial_parameters or {}).get("input_mode") == "video"
        self._cloud_relay: CloudRelay | None = (
            CloudRelay(cloud_manager, video_mode=video_mode)
            if cloud_manager is not None
            else None
        )

        # Input mode: video waits for frames, text generates immediately
        self._video_mode = video_mode

        # Pipeline processors and IDs (populated by _setup_graph)
        self.pipeline_processors: list[PipelineProcessor] = []
        self.pipeline_ids: list[str] = []

        # Graph support: processors indexed by node_id for per-node routing
        self._processors_by_node_id: dict[str, PipelineProcessor] = {}
        # The processor whose output we read in graph mode (legacy get() path)
        self._sink_processor: PipelineProcessor | None = None

        # Source manager (sources, source queues, hardware input)
        self._source_manager = SourceManager()
        self._source_manager.set_on_frame(self._on_hardware_source_frame)

        # Sink manager (sinks, sink queues, hardware output, recording)
        self.sink_manager = SinkManager()

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
        self._source_manager.start()
        self.sink_manager.start()

        # Process output sink settings from initial parameters
        if "output_sinks" in self.parameters:
            sinks_config = self.parameters.pop("output_sinks")
            self.sink_manager.update_config(
                sinks_config, self._get_pipeline_dimensions()
            )

        # Process generic input source settings.
        # When a graph has source nodes, per-node input setup handles routing,
        # so skip the global input_source mechanism.
        if "input_source" in self.parameters:
            graph_data = self.parameters.get("graph")
            has_graph_sources = False
            if graph_data and isinstance(graph_data, dict):
                has_graph_sources = any(
                    n.get("type") == "source" for n in graph_data.get("nodes", [])
                )
            if has_graph_sources:
                self.parameters.pop("input_source")
            else:
                input_source_config = self.parameters.pop("input_source")
                self._source_manager.update_config(input_source_config)

        # Reset frame counters on start
        self._frames_in = 0
        self._frames_out = 0
        self._last_heartbeat_time = time.time()
        self._playback_ready_emitted = False
        self._stream_start_time = time.monotonic()
        self._last_stats_time = time.time()

        if self._cloud_relay is not None:
            # Cloud mode: frames go to cloud instead of local pipelines
            logger.info("[FRAME-PROCESSOR] Starting in CLOUD mode (cloud)")

            self._cloud_relay.start()

            # Set up per-node input sources and record queues in cloud mode.
            graph_data = self.parameters.get("graph")
            if graph_data and isinstance(graph_data, dict):
                from .graph_schema import GraphConfig

                graph = GraphConfig(**graph_data)
                self._source_manager.setup_multi_sources(graph)
                self.sink_manager.setup_cloud_graph(graph)

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

        # Clean up all outputs (sinks + recording)
        self.sink_manager.stop()

        # Clean up all input sources
        self._source_manager.stop()

        # Clean up cloud relay
        if self._cloud_relay is not None:
            self._cloud_relay.stop()

        # Log final frame stats
        if self._cloud_relay is not None:
            logger.info(
                f"[FRAME-PROCESSOR] Stopped (cloud mode). "
                f"Frames: in={self._frames_in}, to_cloud={self._cloud_relay.frames_to_cloud}, "
                f"from_cloud={self._cloud_relay.frames_from_cloud}, out={self._frames_out}"
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
                    "mode": "cloud" if self._cloud_relay is not None else "local",
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
                "mode": "cloud" if self._cloud_relay is not None else "local",
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

    def _on_hardware_source_frame(
        self, source_node_id: str | None, rgb_frame: np.ndarray
    ) -> None:
        """Callback invoked by SourceManager when a hardware source produces a frame.

        Routes the frame to cloud or to local source queues depending on mode.
        source_node_id is None for the generic (non-graph) input source.
        """
        if self._cloud_relay is not None:
            if source_node_id is not None:
                self._cloud_relay.send_frame_to_source(rgb_frame, source_node_id)
            else:
                self._cloud_relay.send_frame(rgb_frame)
            return

        # Local mode: convert to tensor and route to source queues
        frame_tensor = self._frame_array_to_tensor(rgb_frame)
        if source_node_id is not None:
            self._source_manager.route_frame_to_source(frame_tensor, source_node_id)
        else:
            self._source_manager.route_frame_to_all_sources(frame_tensor)

    def _maybe_emit_frame_heartbeat(self) -> None:
        """Log stats periodically when frames flow (shared by put() paths)."""
        now = time.time()
        if now - self._last_heartbeat_time >= HEARTBEAT_INTERVAL_SECONDS:
            self._log_frame_stats()
            self._last_heartbeat_time = now

    def put(self, frame: VideoFrame) -> bool:
        """Put a frame into the pipeline.

        For single-source graphs, delegates to put_to_source() with the
        sole source node.  For multi-source graphs, callers must use
        put_to_source() directly — this method returns False.
        """
        if not self.running:
            return False

        # Single-source shortcut: delegate to put_to_source
        single_id = self._source_manager.single_source_node_id
        if single_id is not None:
            return self.put_to_source(frame, single_id, count_frame=True)

        if self._cloud_relay is not None:
            self._frames_in += 1
            self._maybe_emit_frame_heartbeat()
            if not self._video_mode:
                return True
            frame_array = frame.to_ndarray(format="rgb24")
            self._cloud_relay.send_frame(frame_array)
            return True

        # Local mode with no source queues
        self._frames_in += 1
        self._maybe_emit_frame_heartbeat()
        return False

    def put_to_source(
        self,
        frame: VideoFrame,
        source_node_id: str,
        *,
        count_frame: bool = True,
    ) -> bool:
        """Route a frame to a specific source node (multi-source)."""
        if not self.running:
            return False

        if count_frame:
            self._frames_in += 1
            self._maybe_emit_frame_heartbeat()

        if self._cloud_relay is not None:
            if not self._video_mode:
                return True
            frame_array = frame.to_ndarray(format="rgb24")
            self._cloud_relay.send_frame_to_source(frame_array, source_node_id)
            return True

        # Local mode: convert and route to source node queues
        frame_tensor = self._frame_array_to_tensor(frame.to_ndarray(format="rgb24"))
        return self._source_manager.route_frame_to_source(frame_tensor, source_node_id)

    def get_from_sink(self, sink_node_id: str) -> torch.Tensor | None:
        """Read a frame from a specific sink node's output queue (multi-sink)."""
        if not self.running:
            return None
        frame = self.sink_manager.get_from_sink(sink_node_id)
        if frame is not None:
            self._frames_out += 1
        return frame

    def get_sink_node_ids(self) -> list[str]:
        """Return the list of sink node IDs available for reading."""
        return self.sink_manager.get_sink_node_ids()

    def get_unhandled_sink_node_ids(self) -> list[str]:
        """Return sink node IDs that don't have their own output sink thread.

        These sinks need external draining (e.g. by the headless consumer)
        to prevent their queues from filling up and stalling the pipeline.
        """
        return self.sink_manager.get_unhandled_sink_node_ids()

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        # Get frame based on mode
        frame: torch.Tensor | None = None

        if self._cloud_relay is not None:
            frame = self._cloud_relay.get_frame()
            if frame is None:
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
        In cloud mode, reads from the CloudRelay audio queue.

        Returns:
            Tuple of (audio_tensor, sample_rate) or (None, None) if no audio available.
            audio_tensor shape: (channels, samples) - typically (2, N) for stereo
        """
        if not self.running:
            return None, None

        if self._cloud_relay is not None:
            return self._cloud_relay.get_audio()

        if self._sink_processor is None:
            return None, None

        try:
            audio, sample_rate = self._sink_processor.audio_output_queue.get_nowait()
            # Pass through flush sentinels (audio=None, sample_rate=-1)
            return audio, sample_rate
        except queue.Empty:
            return None, None

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
        fps = self.sink_manager.get_fps_for_sink(sink_node_id)
        if fps is not None:
            return fps
        return self.get_fps()

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
                    "mode": "cloud" if self._cloud_relay is not None else "local",
                    "ttff_ms": time_to_first_frame_ms,
                },
                connection_info=self.connection_info,
            )
            logger.info(
                f"[FRAME-PROCESSOR] First frame produced, playback ready "
                f"(session={self.session_id}, mode={'cloud' if self._cloud_relay is not None else 'local'}, "
                f"ttff={time_to_first_frame_ms}ms)"
            )

        if self.sink_manager.has_generic_sinks:
            try:
                frame_np = frame.numpy()
                self.sink_manager.fan_out_frame(frame_np)
            except Exception as e:
                logger.error(f"Error enqueueing output sink frame: {e}")

    def _log_frame_stats(self):
        """Log frame processing statistics and emit heartbeat event."""
        now = time.time()
        elapsed = now - self._last_stats_time

        if elapsed > 0:
            fps_in = self._frames_in / elapsed if self._frames_in > 0 else 0
            fps_out = self._frames_out / elapsed if self._frames_out > 0 else 0
            cloud = self._cloud_relay
            pipeline_fps = self.get_fps() if cloud is None else None

            if cloud is not None:
                logger.info(
                    f"[FRAME-PROCESSOR] RELAY MODE | "
                    f"Frames: in={self._frames_in}, to_cloud={cloud.frames_to_cloud}, "
                    f"from_cloud={cloud.frames_from_cloud}, out={self._frames_out} | "
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
                "mode": "cloud" if cloud is not None else "local",
                "frames_in": self._frames_in,
                "frames_out": self._frames_out,
                "fps_in": round(fps_in, 1),
                "fps_out": round(fps_out, 1),
                "elapsed_ms": int(elapsed * 1000),
                "since_last_heartbeat_ms": int(
                    (now - self._last_heartbeat_time) * 1000
                ),
            }
            if cloud is not None:
                heartbeat_metadata["frames_to_cloud"] = cloud.frames_to_cloud
                heartbeat_metadata["frames_from_cloud"] = cloud.frames_from_cloud
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

        cloud = self._cloud_relay
        stats = {
            "frames_in": self._frames_in,
            "frames_out": self._frames_out,
            "elapsed_seconds": elapsed,
            "fps_in": self._frames_in / elapsed if elapsed > 0 else 0,
            "fps_out": self._frames_out / elapsed if elapsed > 0 else 0,
            "pipeline_fps": self.get_fps(),
            "output_sinks": self.sink_manager.get_info(),
            "input_source_enabled": self._source_manager.enabled,
            "input_source_type": self._source_manager.source_type,
            "relay_mode": cloud is not None,
        }

        if cloud is not None:
            stats["frames_to_cloud"] = cloud.frames_to_cloud
            stats["frames_from_cloud"] = cloud.frames_from_cloud

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
            self.sink_manager.update_config(
                sinks_config, self._get_pipeline_dimensions()
            )

        # Handle generic input source settings.
        # Skip when per-node sources or graph source queues are active.
        if "input_source" in parameters:
            if (
                self._source_manager.has_per_node_sources
                or self._source_manager.has_source_queues
            ):
                parameters.pop("input_source")
            else:
                input_source_config = parameters.pop("input_source")
                self._source_manager.update_config(input_source_config)

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
                if self._cloud_relay is not None:
                    self._cloud_relay.video_mode = False
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

        self._sink_processor = graph_run.sink_processor
        self.pipeline_processors = graph_run.processors
        self.pipeline_ids = graph_run.pipeline_ids

        # Delegate queue routing to managers
        self._source_manager.setup_graph_queues(
            source_queues=graph_run.source_queues,
            source_queues_by_node=graph_run.source_queues_by_node,
        )
        self.sink_manager.setup_graph_queues(
            sink_queues_by_node=graph_run.sink_queues_by_node,
            sink_hardware_queues_by_node=graph_run.sink_hardware_queues_by_node,
            sink_processors_by_node=graph_run.sink_processors_by_node,
            record_queues_by_node=graph_run.record_queues_by_node,
        )

        # Index processors by node_id for per-node parameter routing
        for proc in self.pipeline_processors:
            self._processors_by_node_id[proc.node_id] = proc

        # Start all processors
        for processor in self.pipeline_processors:
            processor.start()

        # Set up per-source-node input sources for non-WebRTC sources
        self._source_manager.setup_multi_sources(graph)

        # Set up per-sink-node output sinks for non-WebRTC sinks
        self.sink_manager.setup_multi_sinks(graph, self._get_pipeline_dimensions())

        logger.info(
            f"Created graph with {len(self.pipeline_processors)} processors, "
            f"sink={graph_run.sink_node_id}, "
            f"sources={self._source_manager.get_source_node_ids()}, "
            f"sinks={self.sink_manager.get_sink_node_ids()}, "
            f"records={self.sink_manager.get_record_node_ids()}"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
