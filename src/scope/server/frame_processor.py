import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any

import torch
from aiortc.mediastreams import VideoFrame

from .kafka_publisher import publish_event
from .pipeline_manager import PipelineManager
from .pipeline_processor import PipelineProcessor

if TYPE_CHECKING:
    from scope.core.inputs import InputSource
    from scope.core.outputs import OutputSink

    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

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
    ):
        self.pipeline_manager = pipeline_manager
        self.cloud_manager = cloud_manager

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

        # Pinned memory buffer cache for faster GPU transfers (local mode only)
        self._pinned_buffer_cache = {}
        self._pinned_buffer_lock = threading.Lock()

        # Cloud mode: send frames to cloud instead of local processing
        self._cloud_mode = cloud_manager is not None
        self._cloud_output_queue: queue.Queue = queue.Queue(maxsize=2)
        self._frames_to_cloud = 0
        self._frames_from_cloud = 0

        # Output sinks keyed by type
        self.output_sinks: dict[str, dict] = {}

        self.input_source: InputSource | None = None
        self.input_source_enabled = False
        self.input_source_type = ""
        self.input_source_thread = None

        # Input mode: video waits for frames, text generates immediately
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Pipeline chaining support
        self.pipeline_processors: list[PipelineProcessor] = []
        self.pipeline_ids: list[str] = []

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

        # Process generic input source settings
        if "input_source" in self.parameters:
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

            # Register callback to receive frames from cloud
            if self.cloud_manager:
                self.cloud_manager.add_frame_callback(self._on_frame_from_cloud)

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

        # Local mode: setup pipeline chain
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
            self._setup_pipeline_chain_sync()
        except Exception as e:
            logger.error(f"Pipeline chain setup failed: {e}")
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

        # Stop all pipeline processors
        for processor in self.pipeline_processors:
            processor.stop()

        # Clear pipeline processors
        self.pipeline_processors.clear()

        # Clean up all output sinks
        for sink_type, entry in list(self.output_sinks.items()):
            try:
                entry["queue"].put_nowait(None)
            except queue.Full:
                pass
            thread = entry.get("thread")
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
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

        # Clean up cloud callback in cloud mode
        if self._cloud_mode and self.cloud_manager:
            self.cloud_manager.remove_frame_callback(self._on_frame_from_cloud)

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

    def _get_or_create_pinned_buffer(self, shape):
        """Get or create a reusable pinned memory buffer for the given shape.

        This avoids repeated pinned memory allocations, which are expensive.
        Pinned memory enables faster DMA transfers to GPU.
        """
        with self._pinned_buffer_lock:
            if shape not in self._pinned_buffer_cache:
                self._pinned_buffer_cache[shape] = torch.empty(
                    shape, dtype=torch.uint8, pin_memory=True
                )
            return self._pinned_buffer_cache[shape]

    def put(self, frame: VideoFrame) -> bool:
        if not self.running:
            return False

        self._frames_in += 1

        # Log stats and emit heartbeat every HEARTBEAT_INTERVAL_SECONDS
        now = time.time()
        if now - self._last_heartbeat_time >= HEARTBEAT_INTERVAL_SECONDS:
            self._log_frame_stats()
            self._last_heartbeat_time = now

        if self._cloud_mode:
            # Cloud mode: send frame to cloud (only in video mode)
            # In text mode, cloud generates video from prompts only - no input frames
            if not self._video_mode:
                return True  # Silently ignore frames in text mode
            if self.cloud_manager:
                frame_array = frame.to_ndarray(format="rgb24")
                if self.cloud_manager.send_frame(frame_array):
                    self._frames_to_cloud += 1
                    return True
                else:
                    logger.debug("[FRAME-PROCESSOR] Failed to send frame to cloud")
                    return False
            return False

        # Local mode: put into first processor's input queue
        if self.pipeline_processors:
            first_processor = self.pipeline_processors[0]

            frame_array = frame.to_ndarray(format="rgb24")

            if torch.cuda.is_available():
                shape = frame_array.shape
                pinned_buffer = self._get_or_create_pinned_buffer(shape)
                # Note: We reuse pinned buffers for performance. This assumes the copy_()
                # operation completes before the next frame arrives.
                # In practice, copy_() is very fast (~microseconds) and frames arrive at 60 FPS max
                pinned_buffer.copy_(torch.as_tensor(frame_array, dtype=torch.uint8))
                frame_tensor = pinned_buffer.cuda(non_blocking=True)
            else:
                frame_tensor = torch.as_tensor(frame_array, dtype=torch.uint8)

            frame_tensor = frame_tensor.unsqueeze(0)

            # Put frame into first processor's input queue
            try:
                first_processor.input_queue.put_nowait(frame_tensor)
            except queue.Full:
                # Queue full, drop frame (non-blocking)
                logger.debug("First processor input queue full, dropping frame")
                return False

        return True

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

            last_processor = self.pipeline_processors[-1]
            if not last_processor.output_queue:
                return None

            try:
                frame = last_processor.output_queue.get_nowait()
                # Frame is stored as [1, H, W, C], convert to [H, W, C] for output
                # Move to CPU here for WebRTC streaming (frames stay on GPU between pipeline processors)
                frame = frame.squeeze(0)
                if frame.is_cuda:
                    frame = frame.cpu()
            except queue.Empty:
                return None

        # Common processing for both modes
        self._frames_out += 1

        # Emit playback_ready event on first frame output
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

        # Fan out frame to all active output sinks
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

        return frame

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

    def get_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS.

        Returns the FPS based on how fast frames are produced into the last processor's output queue,
        adjusted for queue fill level to prevent buildup.
        """
        if not self.pipeline_processors:
            return DEFAULT_FPS

        # Get FPS from the last processor in the chain
        last_processor = self.pipeline_processors[-1]
        return last_processor.get_fps()

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

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        # Handle generic output sinks config
        if "output_sinks" in parameters:
            sinks_config = parameters.pop("output_sinks")
            self._update_output_sinks_from_config(sinks_config)

        # Handle generic input source settings
        if "input_source" in parameters:
            input_source_config = parameters.pop("input_source")
            self._update_input_source(input_source_config)

        # Update parameters for all pipeline processors
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
        try:
            entry["queue"].put_nowait(None)
        except queue.Full:
            pass
        thread = entry.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
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
        """Background thread that receives frames from a generic input source."""
        logger.info(f"Input source thread started ({self.input_source_type})")

        target_fps = self.get_fps()
        frame_interval = 1.0 / target_fps
        last_frame_time = 0.0
        frame_count = 0

        while (
            self.running and self.input_source_enabled and self.input_source is not None
        ):
            try:
                current_pipeline_fps = self.get_fps()
                if current_pipeline_fps > 0:
                    target_fps = current_pipeline_fps
                    frame_interval = 1.0 / target_fps

                current_time = time.time()
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                    continue

                rgb_frame = self.input_source.receive_frame(timeout_ms=100)
                if rgb_frame is not None:
                    last_frame_time = time.time()

                    if self._cloud_mode:
                        if self._video_mode and self.cloud_manager:
                            if self.cloud_manager.send_frame(rgb_frame):
                                self._frames_to_cloud += 1
                    elif self.pipeline_processors:
                        first_processor = self.pipeline_processors[0]

                        if torch.cuda.is_available():
                            shape = rgb_frame.shape
                            pinned_buffer = self._get_or_create_pinned_buffer(shape)
                            pinned_buffer.copy_(
                                torch.as_tensor(rgb_frame, dtype=torch.uint8)
                            )
                            frame_tensor = pinned_buffer.cuda(non_blocking=True)
                        else:
                            frame_tensor = torch.as_tensor(rgb_frame, dtype=torch.uint8)

                        frame_tensor = frame_tensor.unsqueeze(0)

                        try:
                            first_processor.input_queue.put_nowait(frame_tensor)
                        except queue.Full:
                            logger.debug(
                                f"First processor input queue full, "
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

    def _setup_pipeline_chain_sync(self):
        """Create pipeline processor chain (synchronous).

        Assumes all pipelines are already loaded by the pipeline manager.
        """
        if not self.pipeline_ids:
            logger.error("No pipeline IDs provided")
            return

        # Create pipeline processors (each creates its own queues)
        for pipeline_id in self.pipeline_ids:
            # Get pipeline instance from manager
            pipeline = self.pipeline_manager.get_pipeline_by_id(pipeline_id)

            # Create processor with its own queues
            processor = PipelineProcessor(
                pipeline=pipeline,
                pipeline_id=pipeline_id,
                initial_parameters=self.parameters.copy(),
                session_id=self.session_id,
                user_id=self.user_id,
                connection_id=self.connection_id,
                connection_info=self.connection_info,
            )

            self.pipeline_processors.append(processor)

        for i in range(1, len(self.pipeline_processors)):
            prev_processor = self.pipeline_processors[i - 1]
            curr_processor = self.pipeline_processors[i]
            prev_processor.set_next_processor(curr_processor)

        # Start all processors
        for processor in self.pipeline_processors:
            processor.start()

        logger.info(
            f"Created pipeline chain with {len(self.pipeline_processors)} processors"
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """
        Check if an error is recoverable (i.e., processing can continue).
        Non-recoverable errors will cause the stream to stop.
        """
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        # Add more non-recoverable error types here as needed
        return True
