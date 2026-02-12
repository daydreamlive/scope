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
    from .cloud_connection import CloudConnectionManager

logger = logging.getLogger(__name__)


# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
DEFAULT_FPS = 30.0  # Default FPS

# Heartbeat interval for stream stats logging and Kafka events
HEARTBEAT_INTERVAL_SECONDS = 10.0


class _SpoutFrame:
    """Lightweight wrapper for Spout frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class FrameProcessor:
    """Processes video frames through pipelines or cloud relay.

    Supports two modes:
    1. Local mode: Frames processed through local GPU pipelines
    2. Cloud mode: Frames sent to cloud for processing

    Spout integration works in both modes.
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

        # Spout integration (works in both local and cloud modes)
        self.spout_sender = None
        self.spout_sender_enabled = False
        self.spout_sender_name = "ScopeSyphonSpoutOut"
        self._frame_spout_count = 0
        self.spout_sender_queue = queue.Queue(
            maxsize=30
        )  # Queue for async Spout sending
        self.spout_sender_thread = None

        # Spout input
        self.spout_receiver = None
        self.spout_receiver_enabled = False
        self.spout_receiver_name = ""
        self.spout_receiver_thread = None

        # Input mode is signaled by the frontend at stream start.
        # This determines whether we wait for video frames or generate immediately.
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Pipeline chaining support (local mode only)
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

        # Process any Spout settings from initial parameters
        if "spout_sender" in self.parameters:
            spout_config = self.parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        if "spout_receiver" in self.parameters:
            spout_config = self.parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

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
            logger.error("No pipeline IDs provided, cannot start")
            self.running = False
            return

        try:
            self._setup_pipeline_chain_sync()
        except Exception as e:
            logger.error(f"Pipeline chain setup failed: {e}")
            self.running = False
            return

        # Register for pipeline unload notifications to release references
        if self.pipeline_manager:
            self.pipeline_manager.register_unload_callback(self._on_pipeline_unloading)

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

        # Unregister pipeline unload callback
        if self.pipeline_manager:
            self.pipeline_manager.unregister_unload_callback(
                self._on_pipeline_unloading
            )

        # Stop all pipeline processors
        for processor in self.pipeline_processors:
            processor.stop()

        # Clear pipeline processors
        self.pipeline_processors.clear()

        # Clear pinned memory buffers
        self._pinned_buffer_cache.clear()

        # Clean up Spout sender
        self.spout_sender_enabled = False
        if self.spout_sender_thread and self.spout_sender_thread.is_alive():
            # Signal thread to stop by putting None in queue
            try:
                self.spout_sender_queue.put_nowait(None)
            except queue.Full:
                pass
            self.spout_sender_thread.join(timeout=2.0)
        if self.spout_sender is not None:
            try:
                self.spout_sender.release()
            except Exception as e:
                logger.error(f"Error releasing Spout sender: {e}")
            self.spout_sender = None

        # Clean up Spout receiver
        self.spout_receiver_enabled = False
        if self.spout_receiver is not None:
            try:
                self.spout_receiver.release()
            except Exception as e:
                logger.error(f"Error releasing Spout receiver: {e}")
            self.spout_receiver = None

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
            # Publish stream_error event
            publish_event(
                event_type="stream_error",
                session_id=self.session_id,
                connection_id=self.connection_id,
                pipeline_ids=self.pipeline_ids if self.pipeline_ids else None,
                user_id=self.user_id,
                error={
                    "message": error_message,
                    "type": "StreamError",
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

        # Enqueue frame for async Spout sending (non-blocking)
        if self.spout_sender_enabled and self.spout_sender is not None:
            try:
                # Frame is (H, W, C) uint8 [0, 255]
                frame_np = frame.numpy()
                self.spout_sender_queue.put_nowait(frame_np)
            except queue.Full:
                # Queue full, drop frame (non-blocking)
                logger.debug("Spout output queue full, dropping frame")
            except Exception as e:
                logger.error(f"Error enqueueing Spout frame: {e}")

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
            "spout_receiver_enabled": self.spout_receiver_enabled,
            "spout_sender_enabled": self.spout_sender_enabled,
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

    def _on_pipeline_unloading(self, pipeline_id: str):
        """Release pipeline reference when a pipeline is being unloaded."""
        for processor in self.pipeline_processors:
            if processor.pipeline_id == pipeline_id:
                processor.stop()
                processor.release_pipeline()

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        # Handle Spout output settings
        if "spout_sender" in parameters:
            spout_config = parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        # Handle Spout input settings
        if "spout_receiver" in parameters:
            spout_config = parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        # Update parameters for all pipeline processors
        for processor in self.pipeline_processors:
            processor.update_parameters(parameters)

        # Update local parameters
        self.parameters = {**self.parameters, **parameters}

        return True

    def _update_spout_sender(self, config: dict):
        """Update Spout output configuration."""
        logger.info(f"Spout output config received: {config}")

        enabled = config.get("enabled", False)
        sender_name = config.get("name", "ScopeSyphonSpoutOut")

        # Get dimensions from active pipeline
        width, height = self._get_pipeline_dimensions()

        logger.info(
            f"Spout output: enabled={enabled}, name={sender_name}, size={width}x{height}"
        )

        # Lazy import SpoutSender
        try:
            from scope.server.spout import SpoutSender
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        if enabled and not self.spout_sender_enabled:
            # Enable Spout output
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_enabled = True
                    self.spout_sender_name = sender_name
                    # Start background thread for async sending
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(f"Spout output enabled: '{sender_name}'")
                else:
                    logger.error("Failed to create Spout sender")
                    self.spout_sender = None
            except Exception as e:
                logger.error(f"Error creating Spout sender: {e}")
                self.spout_sender = None

        elif not enabled and self.spout_sender_enabled:
            # Disable Spout output
            if self.spout_sender is not None:
                self.spout_sender.release()
                self.spout_sender = None
            self.spout_sender_enabled = False
            logger.info("Spout output disabled")

        elif enabled and (
            sender_name != self.spout_sender_name
            or (
                self.spout_sender
                and (
                    self.spout_sender.width != width
                    or self.spout_sender.height != height
                )
            )
        ):
            # Name or dimensions changed, recreate sender
            if self.spout_sender is not None:
                self.spout_sender.release()
            try:
                self.spout_sender = SpoutSender(sender_name, width, height)
                if self.spout_sender.create():
                    self.spout_sender_name = sender_name
                    # Ensure output thread is running
                    if (
                        self.spout_sender_thread is None
                        or not self.spout_sender_thread.is_alive()
                    ):
                        self.spout_sender_thread = threading.Thread(
                            target=self._spout_sender_loop, daemon=True
                        )
                        self.spout_sender_thread.start()
                    logger.info(
                        f"Spout output updated: '{sender_name}' ({width}x{height})"
                    )
                else:
                    logger.error("Failed to recreate Spout sender")
                    self.spout_sender = None
                    self.spout_sender_enabled = False
            except Exception as e:
                logger.error(f"Error recreating Spout sender: {e}")
                self.spout_sender = None
                self.spout_sender_enabled = False

    def _update_spout_receiver(self, config: dict):
        """Update Spout input configuration."""
        enabled = config.get("enabled", False)
        sender_name = config.get("name", "")

        # Lazy import SpoutReceiver
        try:
            from scope.server.spout import SpoutReceiver
        except ImportError:
            if enabled:
                logger.warning("Spout module not available on this platform")
            return

        if enabled and not self.spout_receiver_enabled:
            # Enable Spout input
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    # Start receiving thread
                    self.spout_receiver_thread = threading.Thread(
                        target=self._spout_receiver_loop, daemon=True
                    )
                    self.spout_receiver_thread.start()
                    logger.info(f"Spout input enabled: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to create Spout receiver")
                    self.spout_receiver = None
            except Exception as e:
                logger.error(f"Error creating Spout receiver: {e}")
                self.spout_receiver = None

        elif not enabled and self.spout_receiver_enabled:
            # Disable Spout input
            self.spout_receiver_enabled = False
            if self.spout_receiver is not None:
                self.spout_receiver.release()
                self.spout_receiver = None
            logger.info("Spout input disabled")

        elif enabled and sender_name != self.spout_receiver_name:
            # Name changed, recreate receiver
            self.spout_receiver_enabled = False
            if self.spout_receiver is not None:
                self.spout_receiver.release()
            try:
                self.spout_receiver = SpoutReceiver(sender_name, 512, 512)
                if self.spout_receiver.create():
                    self.spout_receiver_enabled = True
                    self.spout_receiver_name = sender_name
                    # Restart receiving thread if not running
                    if (
                        self.spout_receiver_thread is None
                        or not self.spout_receiver_thread.is_alive()
                    ):
                        self.spout_receiver_thread = threading.Thread(
                            target=self._spout_receiver_loop, daemon=True
                        )
                        self.spout_receiver_thread.start()
                    logger.info(f"Spout input changed to: '{sender_name or 'any'}'")
                else:
                    logger.error("Failed to recreate Spout receiver")
                    self.spout_receiver = None
            except Exception as e:
                logger.error(f"Error recreating Spout receiver: {e}")
                self.spout_receiver = None

    def _spout_sender_loop(self):
        """Background thread that sends frames to Spout asynchronously."""
        logger.info("Spout output thread started")
        frame_count = 0

        while (
            self.running and self.spout_sender_enabled and self.spout_sender is not None
        ):
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_np = self.spout_sender_queue.get(timeout=0.1)
                    # None is a sentinel value to stop the thread
                    if frame_np is None:
                        break
                except queue.Empty:
                    continue

                # Send frame to Spout
                success = self.spout_sender.send(frame_np)
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(
                        f"Spout sent frame {frame_count}, "
                        f"shape={frame_np.shape}, success={success}"
                    )
                self._frame_spout_count = frame_count

            except Exception as e:
                logger.error(f"Error in Spout output loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout output thread stopped after {frame_count} frames")

    def _spout_receiver_loop(self):
        """Background thread that receives frames from Spout and adds to buffer."""
        logger.info("Spout input thread started")

        # Initial target frame rate
        target_fps = self.get_fps()
        frame_interval = 1.0 / target_fps
        last_frame_time = 0.0
        frame_count = 0

        while (
            self.running
            and self.spout_receiver_enabled
            and self.spout_receiver is not None
        ):
            try:
                # Update target FPS dynamically from pipeline performance
                current_pipeline_fps = self.get_fps()
                if current_pipeline_fps > 0:
                    target_fps = current_pipeline_fps
                    frame_interval = 1.0 / target_fps

                current_time = time.time()

                # Frame rate limiting - don't receive faster than target FPS
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                    continue

                # Receive directly as RGB (avoids extra copy from RGBA slice)
                rgb_frame = self.spout_receiver.receive(as_rgb=True)
                if rgb_frame is not None:
                    last_frame_time = time.time()

                    if self._cloud_mode:
                        # Cloud mode: send Spout frames to cloud (only in video mode)
                        # In text mode, cloud generates video from prompts only - no input frames
                        if self._video_mode and self.cloud_manager:
                            if self.cloud_manager.send_frame(rgb_frame):
                                self._frames_to_cloud += 1
                    elif self.pipeline_processors:
                        # Local mode: put into first processor's input queue
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
                                "First processor input queue full, dropping Spout frame"
                            )

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"Spout input received {frame_count} frames")
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in Spout input loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout input thread stopped after {frame_count} frames")

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
