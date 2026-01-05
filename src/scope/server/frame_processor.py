import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch
from aiortc.mediastreams import VideoFrame

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 5  # Increased from 3 to handle faster production rates

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
DEFAULT_FPS = 30.0  # Default FPS
SLEEP_TIME = 0.01

# Input FPS measurement constants
INPUT_FPS_SAMPLE_SIZE = 30  # Number of frame intervals to track
INPUT_FPS_MIN_SAMPLES = 5  # Minimum samples needed before using input FPS


class _SpoutFrame:
    """Lightweight wrapper for Spout frames to match VideoFrame interface."""

    __slots__ = ["_data"]

    def __init__(self, data):
        self._data = data

    def to_ndarray(self, format="rgb24"):
        return self._data


class _FrameWithTimestamp:
    """Wrapper for frames with input timestamp for latency tracking."""

    __slots__ = ["frame", "input_timestamp"]

    def __init__(self, frame, input_timestamp: float):
        self.frame = frame
        self.input_timestamp = input_timestamp

    def to_ndarray(self, format="rgb24"):
        """Delegate to underlying frame's to_ndarray method."""
        return self.frame.to_ndarray(format=format)


class FrameProcessor:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        max_output_queue_size: int = 60,  # Increased from 8 to handle burst production
        max_parameter_queue_size: int = 8,
        max_buffer_size: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        self.pipeline_manager = pipeline_manager

        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.frame_buffer_lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=max_output_queue_size)

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=max_parameter_queue_size)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Callback to notify when frame processor stops
        self.notification_callback = notification_callback

        # FPS tracking variables
        self.processing_time_per_frame = deque(
            maxlen=2
        )  # Keep last 2 processing_time/num_frames values for averaging
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        self.min_fps = MIN_FPS
        self.max_fps = MAX_FPS
        self.current_pipeline_fps = DEFAULT_FPS
        self.fps_lock = threading.Lock()  # Lock for thread-safe FPS updates

        # Input FPS tracking variables
        self.input_frame_times = deque(maxlen=INPUT_FPS_SAMPLE_SIZE)
        self.current_input_fps = DEFAULT_FPS
        self.last_input_fps_update = time.time()
        self.input_fps_lock = threading.Lock()

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

        # Async preprocessing state
        self._preprocessor_types: list[str] = []  # Current preprocessor types (list)
        self._latest_preprocessor_results: dict[str, Any] = {}  # Dict of preprocessor_type -> cached result
        self._preprocessor_submit_chunk_id = 0  # Counter for submitted chunks
        self._last_preprocessor_submit_time = 0.0
        self._preprocessor_result_lock = threading.Lock()

        # End-to-end latency tracking
        self._latency_samples = deque(maxlen=30)  # Keep last 30 latency measurements
        self._latency_lock = threading.Lock()
        self._current_average_latency = 0.0  # Current average latency in ms

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        # Process any Spout settings from initial parameters
        if "spout_sender" in self.parameters:
            spout_config = self.parameters.pop("spout_sender")
            self._update_spout_sender(spout_config)

        if "spout_receiver" in self.parameters:
            spout_config = self.parameters.pop("spout_receiver")
            self._update_spout_receiver(spout_config)

        # Load preprocessors from initial parameters if specified
        if "preprocessor_types" in self.parameters:
            preprocessor_types = self.parameters.get("preprocessor_types")
            if preprocessor_types:
                self.pipeline_manager._load_preprocessors(
                    preprocessor_types,
                    encoder="vits" if "depthanything" in preprocessor_types else None
                )
                self._preprocessor_types = preprocessor_types
        elif "preprocessor_type" in self.parameters:
            # Backward compatibility: handle single preprocessor_type
            preprocessor_type = self.parameters.get("preprocessor_type")
            if preprocessor_type is not None:
                preprocessor_types = [preprocessor_type]
                self.pipeline_manager._load_preprocessors(
                    preprocessor_types,
                    encoder="vits" if preprocessor_type == "depthanything" else None
                )
                self._preprocessor_types = preprocessor_types

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("FrameProcessor started")

    def stop(self, error_message: str = None):
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            # Don't join if we're calling stop() from within the worker thread
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        with self.frame_buffer_lock:
            self.frame_buffer.clear()

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

        # Clear input frame times
        with self.input_fps_lock:
            self.input_frame_times.clear()

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

        # Track input frame timestamp for FPS measurement
        self.track_input_frame()

        # Wrap frame with timestamp for end-to-end latency tracking
        input_timestamp = time.time()
        frame_with_timestamp = _FrameWithTimestamp(frame, input_timestamp)

        with self.frame_buffer_lock:
            self.frame_buffer.append(frame_with_timestamp)
            return True

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        try:
            queue_item = self.output_queue.get_nowait()

            # Unwrap frame and timestamp if wrapped for latency tracking
            if isinstance(queue_item, tuple) and len(queue_item) == 2:
                frame, input_timestamp = queue_item
                # Calculate end-to-end latency from camera input to playback output
                # This includes: buffer wait + preprocessing + pipeline processing + queue wait
                output_timestamp = time.time()
                latency_ms = (output_timestamp - input_timestamp) * 1000.0

                # Track latency samples
                with self._latency_lock:
                    self._latency_samples.append(latency_ms)
                    if len(self._latency_samples) >= 10:
                        # Log average latency every 10 frames
                        avg_latency = sum(self._latency_samples) / len(self._latency_samples)
                        min_latency = min(self._latency_samples)
                        max_latency = max(self._latency_samples)
                        logger.info(
                            f"[Latency] End-to-end: avg={avg_latency:.1f}ms, "
                            f"min={min_latency:.1f}ms, max={max_latency:.1f}ms "
                            f"(sample_size={len(self._latency_samples)})"
                        )
                        # Update current average latency for API access
                        self._current_average_latency = avg_latency
                        self._latency_samples.clear()
                    elif len(self._latency_samples) > 0:
                        # Update average even if we don't have 10 samples yet
                        self._current_average_latency = sum(self._latency_samples) / len(self._latency_samples)
            else:
                # Backward compatibility: handle old format without timestamps
                frame = queue_item

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

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS"""
        with self.fps_lock:
            return self.current_pipeline_fps

    def get_average_latency(self) -> float:
        """Get the current average end-to-end latency in milliseconds.

        Returns 0.0 if no latency measurements are available yet.
        """
        with self._latency_lock:
            return self._current_average_latency

    def get_output_fps(self) -> float:
        """Get the output FPS that frames should be sent at.

        Returns the minimum of input FPS and pipeline FPS to ensure:
        1. We don't send frames faster than they were captured (maintains temporal accuracy)
        2. We don't try to output faster than the pipeline can produce (prevents frame starvation)
        """
        input_fps = self._get_input_fps()
        pipeline_fps = self.get_current_pipeline_fps()

        if input_fps is None:
            return pipeline_fps

        # Use minimum to respect both input rate and pipeline capacity
        output_fps = min(input_fps, pipeline_fps)

        # Log FPS breakdown occasionally for debugging
        if hasattr(self, '_last_fps_log_time'):
            if time.time() - self._last_fps_log_time > 5.0:
                logger.info(
                    f"[FPS] Input: {input_fps:.1f}, Pipeline: {pipeline_fps:.1f}, "
                    f"Output: {output_fps:.1f}"
                )
                self._last_fps_log_time = time.time()
        else:
            self._last_fps_log_time = time.time()

        return output_fps

    def _get_input_fps(self) -> float | None:
        """Get the current measured input FPS.

        Returns the measured input FPS if enough samples are available,
        otherwise returns None to indicate fallback should be used.
        """
        with self.input_fps_lock:
            if len(self.input_frame_times) < INPUT_FPS_MIN_SAMPLES:
                return None
            return self.current_input_fps

    def _calculate_input_fps(self):
        """Calculate and update input FPS from recent frame timestamps.

        Uses the same time-based update logic as pipeline FPS for consistency.
        Only updates if enough time has passed since the last update.
        """
        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_input_fps_update >= self.fps_update_interval:
            with self.input_fps_lock:
                if len(self.input_frame_times) >= INPUT_FPS_MIN_SAMPLES:
                    # Calculate FPS from frame intervals
                    times = list(self.input_frame_times)
                    if len(times) >= 2:
                        # Time span from first to last frame
                        time_span = times[-1] - times[0]
                        if time_span > 0:
                            # FPS = (number of intervals) / time_span
                            num_intervals = len(times) - 1
                            estimated_fps = num_intervals / time_span

                            # Clamp to reasonable bounds (same as pipeline FPS)
                            estimated_fps = max(
                                self.min_fps, min(self.max_fps, estimated_fps)
                            )
                            self.current_input_fps = estimated_fps

            self.last_input_fps_update = current_time

    def track_input_frame(self):
        """Track timestamp of an incoming frame for FPS measurement"""
        with self.input_fps_lock:
            self.input_frame_times.append(time.time())

        # Update input FPS calculation using same logic as pipeline FPS
        self._calculate_input_fps()

    def _calculate_pipeline_fps(self, start_time: float, num_frames: int):
        """Calculate FPS based on processing time and number of frames created"""
        processing_time = time.time() - start_time
        if processing_time <= 0 or num_frames <= 0:
            return

        # Store processing time per frame for averaging
        time_per_frame = processing_time / num_frames
        self.processing_time_per_frame.append(time_per_frame)

        # Update FPS if enough time has passed
        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.processing_time_per_frame) >= 1:
                # Calculate average processing time per frame
                avg_time_per_frame = sum(self.processing_time_per_frame) / len(
                    self.processing_time_per_frame
                )

                # Calculate FPS: 1 / average_time_per_frame
                # This gives us the actual frames per second output
                with self.fps_lock:
                    current_fps = self.current_pipeline_fps
                estimated_fps = (
                    1.0 / avg_time_per_frame if avg_time_per_frame > 0 else current_fps
                )

                # Clamp to reasonable bounds
                estimated_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
                with self.fps_lock:
                    self.current_pipeline_fps = estimated_fps

            self.last_fps_update = current_time

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

        # Put new parameters in queue (replace any pending update)
        try:
            # Add new update
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.info("Parameter queue full, dropping parameter update")
            return False

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
        target_fps = self.get_current_pipeline_fps()
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
                current_pipeline_fps = self.get_current_pipeline_fps()
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
                    spout_frame = _SpoutFrame(rgb_frame)

                    # Wrap with timestamp for latency tracking
                    input_timestamp = time.time()
                    frame_with_timestamp = _FrameWithTimestamp(spout_frame, input_timestamp)

                    with self.frame_buffer_lock:
                        self.frame_buffer.append(frame_with_timestamp)

                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"Spout input received {frame_count} frames")
                else:
                    time.sleep(0.001)  # Small sleep when no frame available

            except Exception as e:
                logger.error(f"Error in Spout input loop: {e}")
                time.sleep(0.01)

        logger.info(f"Spout input thread stopped after {frame_count} frames")

    def worker_loop(self):
        logger.info("Worker thread started")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except PipelineNotAvailableException as e:
                logger.debug(f"Pipeline temporarily unavailable: {e}")
                # Flush frame buffer to prevent buildup
                with self.frame_buffer_lock:
                    if self.frame_buffer:
                        logger.debug(
                            f"Flushing {len(self.frame_buffer)} frames due to pipeline unavailability"
                        )
                        self.frame_buffer.clear()
                continue
            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(f"Error in worker loop: {e}")
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop: {e}, stopping frame processor"
                    )
                    self.stop(error_message=str(e))
                    break
        logger.info("Worker thread stopped")

    def process_chunk(self):
        start_time = time.time()
        try:
            # Check if there are new parameters
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

                # Handle preprocessor type changes
                if "preprocessor_types" in new_parameters:
                    new_preprocessor_types = new_parameters.get("preprocessor_types") or []
                    if set(new_preprocessor_types) != set(self._preprocessor_types):
                        # Update preprocessors
                        self.pipeline_manager._load_preprocessors(
                            new_preprocessor_types,
                            encoder="vits" if "depthanything" in new_preprocessor_types else None
                        )
                        self._preprocessor_types = new_preprocessor_types
                        with self._preprocessor_result_lock:
                            self._latest_preprocessor_results.clear()
                elif "preprocessor_type" in new_parameters:
                    # Backward compatibility: handle single preprocessor_type
                    new_preprocessor_type = new_parameters.get("preprocessor_type")
                    new_preprocessor_types = [new_preprocessor_type] if new_preprocessor_type else []
                    if set(new_preprocessor_types) != set(self._preprocessor_types):
                        self.pipeline_manager._load_preprocessors(
                            new_preprocessor_types,
                            encoder="vits" if new_preprocessor_type == "depthanything" else None
                        )
                        self._preprocessor_types = new_preprocessor_types
                        with self._preprocessor_result_lock:
                            self._latest_preprocessor_results.clear()

                # Merge new parameters with existing ones to preserve any missing keys
                self.parameters = {**self.parameters, **new_parameters}
        except queue.Empty:
            pass

        # Get the current pipeline using sync wrapper
        pipeline = self.pipeline_manager.get_pipeline()

        # Pause or resume the processing
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            self.paused = paused
        if self.paused:
            # Sleep briefly to avoid busy waiting
            self.shutdown_event.wait(SLEEP_TIME)
            return

        # prepare() will handle any required preparation based on parameters internally
        reset_cache = self.parameters.pop("reset_cache", None)

        # Pop lora_scales to prevent re-processing on every frame
        lora_scales = self.parameters.pop("lora_scales", None)

        # Clear output buffer queue when reset_cache is requested to prevent old frames
        if reset_cache:
            logger.info("Clearing output buffer queue due to reset_cache request")
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break

        requirements = None
        if hasattr(pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected.
                # This allows resolve_input_mode() to detect video mode correctly.
                prepare_params["video"] = True  # Placeholder, actual data passed later
            requirements = pipeline.prepare(
                **prepare_params,
            )

        video_input = None
        chunk_input_timestamp = None
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.frame_buffer_lock:
                if not self.frame_buffer or len(self.frame_buffer) < current_chunk_size:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                prepare_start = time.time()
                video_input, chunk_input_timestamp = self.prepare_chunk(current_chunk_size)
                prepare_time = time.time() - prepare_start
                if prepare_time > 0.01:
                    logger.debug(f"[Overhead] prepare_chunk: {prepare_time*1000:.1f}ms")
        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass lora_scales only when present (one-time update)
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Route video input based on VACE status and preprocessing
            if video_input is not None:
                vace_enabled = getattr(pipeline, "vace_enabled", False)
                # Get preprocessor types (support both new list format and old single value)
                preprocessor_types = self.parameters.get("preprocessor_types")
                if preprocessor_types is None:
                    # Backward compatibility: check for single preprocessor_type
                    preprocessor_type = self.parameters.get("preprocessor_type")
                    preprocessor_types = [preprocessor_type] if preprocessor_type else []

                # Handle preprocessing (all preprocessors use the same async interface)
                if preprocessor_types:
                    async_preprocessors = self.pipeline_manager.async_preprocessors

                    # Verify all preprocessors are running
                    for preprocessor_type in preprocessor_types:
                        if preprocessor_type not in async_preprocessors:
                            raise RuntimeError(
                                f"Preprocessor {preprocessor_type} is enabled but not loaded"
                            )
                        client = async_preprocessors[preprocessor_type]
                        if not client.is_running():
                            raise RuntimeError(
                                f"Preprocessor {preprocessor_type} is enabled but not running"
                            )

                    # === ASYNC PREPROCESSING ===
                    width, height = self._get_pipeline_dimensions()

                    # Fast path for single preprocessor (optimized, matches old behavior)
                    if len(preprocessor_types) == 1:
                        preprocessor_type = preprocessor_types[0]
                        async_preprocessor_client = async_preprocessors[preprocessor_type]

                        # === ASYNC PREPROCESSING ===
                        # Track preprocessor latency
                        preprocessor_start_time = time.time()

                        # Only submit new frames if we don't have a recent cached result
                        # This reduces overhead (tensor→numpy conversion) and GPU contention
                        # Since preprocessor runs faster than pipeline, we can skip some submissions
                        should_submit = True
                        with self._preprocessor_result_lock:
                            if self._latest_preprocessor_results.get(preprocessor_type) is not None:
                                # Skip if we have a cached result and buffer has more results pending
                                if async_preprocessor_client.get_buffer_size() > 0:
                                    should_submit = False

                        submit_time = 0.0
                        if should_submit:
                            # Submit current frames for processing (non-blocking)
                            preprocessor_submit_time = time.time()
                            async_preprocessor_client.submit_frames(
                                video_input,
                                target_height=height,
                                target_width=width,
                            )
                            submit_time = time.time() - preprocessor_submit_time
                            if submit_time > 0.05:
                                logger.debug(f"[Latency] Preprocessor {preprocessor_type} submit: {submit_time*1000:.1f}ms")

                        # Get latest available (non-blocking)
                        get_result_start = time.time()
                        preprocessor_result = async_preprocessor_client.get_latest_result()
                        get_result_time = time.time() - get_result_start

                        if preprocessor_result is not None:
                            # Cache the result for future use
                            with self._preprocessor_result_lock:
                                self._latest_preprocessor_results[preprocessor_type] = preprocessor_result.data

                            # Track processing latency from result metadata if available
                            if hasattr(preprocessor_result, 'processing_time'):
                                processing_time = preprocessor_result.processing_time
                                logger.info(
                                    f"[Latency] Preprocessor {preprocessor_type}: "
                                    f"submit={submit_time*1000:.1f}ms, "
                                    f"process={processing_time*1000:.1f}ms, "
                                    f"get_result={get_result_time*1000:.1f}ms"
                                )

                        # Use cached preprocessor result (may be from previous chunk)
                        with self._preprocessor_result_lock:
                            preprocessed_input = self._latest_preprocessor_results.get(preprocessor_type)

                        gpu_transfer_time = 0.0
                        if preprocessed_input is not None:
                            # Move to correct device and dtype
                            # non_blocking=True since memory is pinned in receiver thread
                            gpu_start = time.time()
                            preprocessed_input = preprocessed_input.to(
                                device=torch.device("cuda"),
                                dtype=torch.bfloat16,
                                non_blocking=True,
                            )
                            gpu_transfer_time = time.time() - gpu_start

                            # Log total preprocessor latency
                            total_preprocessor_time = time.time() - preprocessor_start_time
                            logger.info(
                                f"[Latency] Preprocessor {preprocessor_type} total: {total_preprocessor_time*1000:.1f}ms "
                                f"(submit={submit_time*1000:.1f}ms, gpu_transfer={gpu_transfer_time*1000:.1f}ms)"
                            )

                            logger.debug(
                                f"Using async {preprocessor_type} preprocessor, shape: {preprocessed_input.shape}"
                            )

                            # Use preprocessed input
                            if vace_enabled:
                                call_params["vace_input_frames"] = preprocessed_input
                            else:
                                call_params["video"] = preprocessed_input
                        else:
                            # No preprocessor result available yet, fall back to video input
                            logger.debug(
                                "No async preprocessor result available yet, using video input"
                            )
                            if vace_enabled:
                                call_params["vace_input_frames"] = video_input
                            else:
                                call_params["video"] = video_input

                    else:
                        # === SEQUENTIAL ASYNC PREPROCESSING (multiple preprocessors) ===
                        # Preprocessors run sequentially: output of first becomes input to second, etc.
                        # Start with original video input
                        current_input = video_input
                        preprocessed_input = None

                        # Track total preprocessing time for sequential preprocessors
                        sequential_preprocessor_start = time.time()

                        # Process each preprocessor in order
                        for idx, preprocessor_type in enumerate(preprocessor_types):
                            client = async_preprocessors[preprocessor_type]
                            preprocessor_iter_start = time.time()

                            # Check if we have a cached result for this preprocessor
                            cache_key = f"{preprocessor_type}_{idx}"
                            with self._preprocessor_result_lock:
                                cached_result = self._latest_preprocessor_results.get(cache_key)

                            # Determine if we should submit
                            should_submit = True
                            if cached_result is not None:
                                # Skip if buffer is full (preprocessor is ahead)
                                if client.get_buffer_size() > client.result_buffer_size // 2:
                                    should_submit = False

                            submit_time = 0.0
                            if should_submit:
                                # Convert current_input to numpy format for submission
                                # current_input is list of tensors [(1, H, W, C), ...] or numpy array
                                if isinstance(current_input, list):
                                    # Convert list of tensors to numpy array [F, H, W, C]
                                    stacked = torch.cat(current_input, dim=0)  # [F, H, W, C]
                                    input_np = stacked.cpu().numpy()
                                elif isinstance(current_input, torch.Tensor):
                                    # Tensor might be [1, C, T, H, W] or [T, H, W, C]
                                    if current_input.dim() == 5:
                                        # [1, C, T, H, W] -> [T, H, W, C]
                                        input_np = current_input.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
                                    else:
                                        input_np = current_input.cpu().numpy()
                                else:
                                    input_np = current_input

                                # Ensure uint8 format [F, H, W, C]
                                if input_np.dtype != np.uint8:
                                    if input_np.max() <= 1.0:
                                        input_np = (input_np * 255).astype(np.uint8)
                                    else:
                                        input_np = input_np.astype(np.uint8)

                                # Submit to preprocessor
                                preprocessor_submit_time = time.time()
                                client.submit_frames(
                                    input_np,
                                    target_height=height,
                                    target_width=width,
                                )
                                submit_time = time.time() - preprocessor_submit_time
                                if submit_time > 0.05:
                                    logger.debug(f"[Latency] Preprocessor {preprocessor_type} submit: {submit_time*1000:.1f}ms")

                            # Get result (non-blocking, use cached if available)
                            get_result_start = time.time()
                            result = client.get_latest_result()
                            get_result_time = time.time() - get_result_start

                            if result is not None:
                                # Cache the result
                                with self._preprocessor_result_lock:
                                    self._latest_preprocessor_results[cache_key] = result.data
                                cached_result = result.data

                                # Track processing latency from result metadata if available
                                if hasattr(result, 'processing_time'):
                                    processing_time = result.processing_time
                                    logger.info(
                                        f"[Latency] Preprocessor {preprocessor_type} ({idx+1}/{len(preprocessor_types)}): "
                                        f"submit={submit_time*1000:.1f}ms, "
                                        f"process={processing_time*1000:.1f}ms, "
                                        f"get_result={get_result_time*1000:.1f}ms"
                                    )

                            gpu_transfer_time = 0.0
                            if cached_result is not None:
                                # Move to GPU
                                gpu_start = time.time()
                                result_tensor = cached_result.to(
                                    device=torch.device("cuda"),
                                    dtype=torch.bfloat16,
                                    non_blocking=True,
                                )
                                gpu_transfer_time = time.time() - gpu_start

                                # Log per-preprocessor latency
                                preprocessor_iter_time = time.time() - preprocessor_iter_start
                                logger.info(
                                    f"[Latency] Preprocessor {preprocessor_type} ({idx+1}/{len(preprocessor_types)}) "
                                    f"total: {preprocessor_iter_time*1000:.1f}ms "
                                    f"(submit={submit_time*1000:.1f}ms, gpu_transfer={gpu_transfer_time*1000:.1f}ms)"
                                )

                                # Store final result for pipeline (keep tensor format)
                                preprocessed_input = result_tensor

                                # Only convert to numpy if there's a next preprocessor
                                if idx < len(preprocessor_types) - 1:
                                    # Convert result tensor [1, C, T, H, W] to numpy format for next preprocessor
                                    # Extract frames: [1, C, T, H, W] -> [T, H, W, C] -> numpy [T, H, W, C]
                                    T = result_tensor.shape[2]
                                    result_tensor_permuted = result_tensor.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]

                                    # Convert from [-1, 1] or [0, 1] range to [0, 255] uint8
                                    # Preprocessor outputs are typically in [-1, 1] for depthanything or [0, 1] for passthrough
                                    if result_tensor_permuted.min() < 0:
                                        # [-1, 1] range -> [0, 1]
                                        result_tensor_permuted = (result_tensor_permuted + 1.0) / 2.0
                                    # [0, 1] -> [0, 255] uint8
                                    result_tensor_permuted = (result_tensor_permuted * 255.0).clamp(0, 255).to(torch.uint8)

                                    # Convert to numpy array [T, H, W, C] for next preprocessor
                                    current_input = result_tensor_permuted.cpu().numpy()

                                logger.debug(
                                    f"Preprocessor {preprocessor_type} ({idx+1}/{len(preprocessor_types)}) output shape: {result_tensor.shape}"
                                )
                            else:
                                # No result available yet, can't continue pipeline
                                logger.debug(
                                    f"No result available yet for preprocessor {preprocessor_type} ({idx+1}/{len(preprocessor_types)}), using video input"
                                )
                                preprocessed_input = None
                                break

                        # Use final preprocessed input if available (inside sequential preprocessor block)
                        if preprocessed_input is not None:
                            # Log total sequential preprocessing time
                            total_sequential_time = time.time() - sequential_preprocessor_start
                            logger.info(
                                f"[Latency] Sequential preprocessors {preprocessor_types} total: {total_sequential_time*1000:.1f}ms"
                            )
                            logger.debug(
                                f"Using sequential preprocessors {preprocessor_types}, final shape: {preprocessed_input.shape}"
                            )
                            if vace_enabled:
                                call_params["vace_input_frames"] = preprocessed_input
                            else:
                                call_params["video"] = preprocessed_input
                        else:
                            # Not all preprocessor results available yet, fall back to video input
                            logger.debug(
                                f"Preprocessor pipeline incomplete, using video input"
                            )
                            if vace_enabled:
                                call_params["vace_input_frames"] = video_input
                            else:
                                call_params["video"] = video_input
                elif vace_enabled:
                    # Regular VACE V2V editing mode: route to vace_input_frames
                    call_params["vace_input_frames"] = video_input
                else:
                    # Normal V2V mode: route to video
                    call_params["video"] = video_input

            # Measure pipeline execution latency
            pipeline_start_time = time.time()
            output = pipeline(**call_params)
            pipeline_latency = time.time() - pipeline_start_time

            # Clear vace_ref_images from parameters after use to prevent sending them on subsequent chunks
            # vace_ref_images should only be sent when explicitly provided in parameter updates
            if (
                "vace_ref_images" in call_params
                and "vace_ref_images" in self.parameters
            ):
                self.parameters.pop("vace_ref_images", None)

            # Clear transition when complete (blocks signal completion via _transition_active)
            # Contract: Modular pipelines manage prompts internally; frame_processor manages lifecycle
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(pipeline, "state"):
                    transition_active = pipeline.state.get("_transition_active", False)

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    self.parameters.pop("transition", None)

            pipeline_end_time = time.time()
            processing_time = pipeline_end_time - start_time
            num_frames = output.shape[0]

            # Calculate overhead (time before pipeline call)
            # start_time is set at beginning of process_chunk
            # pipeline call happens after parameter handling, prepare, and depth processing
            overhead_time = processing_time - pipeline_latency

            logger.info(
                f"[Latency] Pipeline: {pipeline_latency*1000:.1f}ms for {num_frames} frames "
                f"({num_frames / pipeline_latency:.1f} FPS)"
            )
            logger.info(
                f"[Latency] Total chunk: {processing_time*1000:.1f}ms "
                f"(pipeline={pipeline_latency*1000:.1f}ms, overhead={overhead_time*1000:.1f}ms, "
                f"total_fps={num_frames / processing_time:.1f})"
            )

            # Normalize to [0, 255] and convert to uint8
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
                .cpu()
            )

            # Resize output queue to meet target max size
            target_output_queue_max_size = num_frames * OUTPUT_QUEUE_MAX_SIZE_FACTOR
            if self.output_queue.maxsize < target_output_queue_max_size:
                logger.info(
                    f"Increasing output queue size to {target_output_queue_max_size}, current size {self.output_queue.maxsize}, num_frames {num_frames}"
                )

                # Transfer frames from old queue to new queue
                old_queue = self.output_queue
                self.output_queue = queue.Queue(maxsize=target_output_queue_max_size)
                while not old_queue.empty():
                    try:
                        frame = old_queue.get_nowait()
                        self.output_queue.put_nowait(frame)
                    except queue.Empty:
                        break

            # Wrap output frames with input timestamp for end-to-end latency tracking
            # This timestamp represents when the input frame arrived from the camera,
            # and will be used to calculate total latency including:
            # - Buffer wait time
            # - Preprocessing time (DepthAnything, etc.) - included because preprocessing
            #   happens between prepare_chunk() and pipeline execution
            # - Pipeline processing time
            # - Output queue wait time
            # Use current time if no input timestamp (text-to-video mode)
            if chunk_input_timestamp is None:
                chunk_input_timestamp = start_time  # Use chunk start time as fallback

            for frame in output:
                try:
                    # Store frame with input timestamp for latency calculation
                    # This preserves the original camera input timestamp through the entire
                    # processing pipeline (preprocessing + pipeline) for accurate E2E measurement
                    frame_with_latency = (frame, chunk_input_timestamp)
                    self.output_queue.put_nowait(frame_with_latency)
                except queue.Full:
                    logger.warning("Output queue full, dropping processed frame")
                    # Update FPS calculation based on processing time and frame count
                    self._calculate_pipeline_fps(start_time, num_frames)
                    continue

            # Update FPS calculation based on processing time and frame count
            self._calculate_pipeline_fps(start_time, num_frames)
        except Exception as e:
            if self._is_recoverable(e):
                # Handle recoverable errors with full stack trace and continue processing
                logger.error(f"Error processing chunk: {e}", exc_info=True)
            else:
                raise e

        self.is_prepared = True

    def prepare_chunk(self, chunk_size: int) -> tuple[list[torch.Tensor], float]:
        """
        Sample frames uniformly from the buffer, convert them to tensors, and remove processed frames.

        This function implements uniform sampling across the entire buffer to ensure
        temporal coverage of input frames. It samples frames at evenly distributed
        indices and removes all frames up to the last sampled frame to prevent
        buffer buildup.

        Note:
            This function must be called with self.frame_buffer_lock held to ensure
            thread safety. The caller is responsible for acquiring the lock.

        Example:
            With buffer_len=8 and chunk_size=4:
            - step = 8/4 = 2.0
            - indices = [0, 2, 4, 6] (uniformly distributed)
            - Returns frames at positions 0, 2, 4, 6
            - Removes frames 0-6 from buffer (7 frames total)

        Returns:
            Tuple of (list of tensor frames, earliest input timestamp)
            - List of tensor frames, each (1, H, W, C) for downstream preprocess_chunk
            - Earliest input timestamp for latency tracking
        """
        # Calculate uniform sampling step
        step = len(self.frame_buffer) / chunk_size
        # Generate indices for uniform sampling
        indices = [round(i * step) for i in range(chunk_size)]
        # Extract VideoFrames at sampled indices
        video_frames = [self.frame_buffer[i] for i in indices]

        # Extract earliest input timestamp for latency tracking
        # This timestamp represents when the frame arrived from the camera (via put()),
        # and will be preserved through preprocessing and pipeline processing for E2E latency measurement
        earliest_timestamp = min(
            frame.input_timestamp if isinstance(frame, _FrameWithTimestamp) else time.time()
            for frame in video_frames
        )

        # Drop all frames up to and including the last sampled frame
        last_idx = indices[-1]
        for _ in range(last_idx + 1):
            self.frame_buffer.popleft()

        # Convert VideoFrames to tensors
        tensor_frames = []
        for video_frame in video_frames:
            # Unwrap if it's a _FrameWithTimestamp
            actual_frame = video_frame.frame if isinstance(video_frame, _FrameWithTimestamp) else video_frame

            # Convert VideoFrame into (1, H, W, C) tensor on cpu
            # The T=1 dimension is expected by preprocess_chunk which rearranges T H W C -> T C H W
            tensor = (
                torch.from_numpy(actual_frame.to_ndarray(format="rgb24"))
                .float()
                .unsqueeze(0)
            )
            tensor_frames.append(tensor)

        return tensor_frames, earliest_timestamp

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
