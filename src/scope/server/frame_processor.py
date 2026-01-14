import logging
import queue
import threading
import time
from typing import Any

import torch
from aiortc.mediastreams import VideoFrame

from .pipeline_manager import PipelineManager
from .pipeline_processor import PipelineProcessor

logger = logging.getLogger(__name__)


# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
DEFAULT_FPS = 30.0  # Default FPS


class _SpoutFrame:
    """Lightweight wrapper for Spout frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class FrameProcessor:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        max_parameter_queue_size: int = 8,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        self.pipeline_manager = pipeline_manager

        # Current parameters
        self.parameters = initial_parameters or {}

        self.running = False

        # Callback to notify when frame processor stops
        self.notification_callback = notification_callback

        self.paused = False

        # Spout integration
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

        # Pipeline chaining support
        self.pipeline_processors: list[PipelineProcessor] = []
        self.pipeline_ids: list[str] = []

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

        logger.info(
            f"FrameProcessor started with {len(self.pipeline_ids)} pipeline(s): {self.pipeline_ids}"
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

        logger.info("FrameProcessor stopped")

        # Notify callback that frame processor has stopped
        if self.notification_callback:
            try:
                message = {"type": "stream_stopped"}
                if error_message:
                    message["error_message"] = error_message
                self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error in frame processor stop callback: {e}")

    def put(self, frame: VideoFrame) -> bool:
        if not self.running:
            return False

        # Convert VideoFrame to tensor and put into first processor's input queue
        if self.pipeline_processors:
            first_processor = self.pipeline_processors[0]

            # Convert VideoFrame to (H, W, C) uint8 tensor, then to [1, H, W, C]
            frame_array = frame.to_ndarray(format="rgb24")
            frame_tensor = torch.from_numpy(frame_array)
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
        if not self.running or not self.pipeline_processors:
            return None

        # Get frame from last pipeline processor's output queue
        last_processor = self.pipeline_processors[-1]
        if not last_processor.output_queue:
            return None

        try:
            frame = last_processor.output_queue.get_nowait()
            # Frame is stored as [1, H, W, C], convert to [H, W, C] for output
            frame = frame.squeeze(0)

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
        except queue.Empty:
            return None

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

                    # Convert to tensor and put into first processor's input queue
                    if self.pipeline_processors:
                        first_processor = self.pipeline_processors[0]
                        frame_tensor = torch.from_numpy(rgb_frame)
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
