import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any

import torch
from aiortc.mediastreams import VideoFrame

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)

# Flag to enable async depth preprocessing (runs in separate process)
ASYNC_DEPTH_PREPROCESSING = True

# Flag to benchmark depth preprocessing only (skip main pipeline)
# Set DEPTH_BENCHMARK_MODE=True environment variable to measure depth FPS without V2V processing
DEPTH_BENCHMARK_MODE = os.environ.get("DEPTH_BENCHMARK_MODE", "").lower() in ("true", "1", "yes")


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

        # Async depth preprocessing state
        self._async_depth_enabled = False
        self._latest_depth_result = None  # Cached depth result for use in pipeline
        self._depth_submit_chunk_id = 0  # Counter for submitted chunks
        self._last_depth_submit_time = 0.0
        self._depth_result_lock = threading.Lock()

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

        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            return True

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        try:
            frame = self.output_queue.get_nowait()
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

                    with self.frame_buffer_lock:
                        self.frame_buffer.append(spout_frame)

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
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.frame_buffer_lock:
                if not self.frame_buffer or len(self.frame_buffer) < current_chunk_size:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                prepare_start = time.time()
                video_input = self.prepare_chunk(current_chunk_size)
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

            # Route video input based on VACE status and depth preprocessing
            if video_input is not None:
                vace_enabled = getattr(pipeline, "vace_enabled", False)
                depth_preprocessor_enabled = self.parameters.get(
                    "depth_preprocessor", False
                )
                depth_mode = self.parameters.get(
                    "depth_preprocessor_mode", "v2v_depth"
                )

                # Handle depth preprocessing
                if depth_preprocessor_enabled:
                    # Check for async depth preprocessor first
                    async_depth_client = self.pipeline_manager.async_depth_preprocessor
                    use_async = (
                        ASYNC_DEPTH_PREPROCESSING
                        and async_depth_client is not None
                        and async_depth_client.is_running()
                    )

                    if use_async:
                        # === ASYNC DEPTH PREPROCESSING ===
                        # Provide FPS feedback to throttle depth worker
                        pipeline_fps = self.get_current_pipeline_fps()
                        async_depth_client.set_target_fps(pipeline_fps)

                        # Only submit new frames if we don't have a recent cached result
                        # This reduces overhead (tensor→numpy conversion) and GPU contention
                        # Since depth runs ~5x faster than pipeline, we can skip some submissions
                        should_submit = True
                        with self._depth_result_lock:
                            if self._latest_depth_result is not None:
                                # Skip if we have a cached result and buffer has more results pending
                                if async_depth_client.get_buffer_size() > 0:
                                    should_submit = False

                        if should_submit:
                            # Submit current frames for processing (non-blocking)
                            width, height = self._get_pipeline_dimensions()
                            depth_submit_time = time.time()
                            async_depth_client.submit_frames(
                                video_input,
                                target_height=height,
                                target_width=width,
                            )
                            submit_overhead = time.time() - depth_submit_time
                            if submit_overhead > 0.05:
                                logger.debug(f"[Overhead] Depth submit: {submit_overhead*1000:.1f}ms")

                        # In benchmark mode, wait for actual result to measure true depth FPS
                        if DEPTH_BENCHMARK_MODE:
                            depth_result = async_depth_client.get_depth_result(
                                wait=True, timeout=5.0
                            )
                            if depth_result is not None:
                                depth_latency = time.time() - depth_submit_time
                                num_frames = len(video_input)
                                logger.info(
                                    f"[DEPTH BENCHMARK] Depth round-trip: {num_frames} frames in "
                                    f"{depth_latency:.3f}s ({num_frames / depth_latency:.1f} FPS)"
                                )
                                with self._depth_result_lock:
                                    self._latest_depth_result = depth_result.depth
                        else:
                            # Normal mode: get latest available (non-blocking)
                            depth_result = async_depth_client.get_latest_depth_result()
                            if depth_result is not None:
                                # Cache the result for future use
                                with self._depth_result_lock:
                                    self._latest_depth_result = depth_result.depth

                        # Use cached depth result (may be from previous chunk)
                        with self._depth_result_lock:
                            depth_input = self._latest_depth_result

                        if depth_input is not None:
                            # Move to correct device and dtype
                            # non_blocking=True since memory is pinned in receiver thread
                            gpu_start = time.time()
                            depth_input = depth_input.to(
                                device=torch.device("cuda"),
                                dtype=torch.bfloat16,
                                non_blocking=True,
                            )
                            gpu_time = time.time() - gpu_start
                            if gpu_time > 0.01:
                                logger.debug(f"[Overhead] Depth GPU transfer: {gpu_time*1000:.1f}ms")

                            logger.debug(
                                f"Using async depth (mode={depth_mode}), "
                                f"shape: {depth_input.shape}"
                            )

                            if depth_mode == "depth_only":
                                if vace_enabled:
                                    call_params["vace_input_frames"] = depth_input
                                else:
                                    call_params["video"] = depth_input
                            else:
                                if vace_enabled:
                                    call_params["vace_input_frames"] = depth_input
                                    call_params["video"] = video_input
                                else:
                                    logger.warning(
                                        "V2V + Depth mode requires VACE. "
                                        "Falling back to depth-only (no VACE)."
                                    )
                                    call_params["video"] = depth_input
                        else:
                            # No depth result available yet, fall back to video input
                            logger.debug(
                                "No async depth result available yet, using video input"
                            )
                            if vace_enabled:
                                call_params["vace_input_frames"] = video_input
                            else:
                                call_params["video"] = video_input
                    else:
                        # === SYNC DEPTH PREPROCESSING (fallback) ===
                        depth_preprocessor = self.pipeline_manager.depth_preprocessor
                        if depth_preprocessor is not None:
                            # Apply depth preprocessing to video input
                            depth_input = self._apply_depth_preprocessing(
                                video_input, depth_preprocessor
                            )
                            logger.info(
                                f"Applied depth preprocessing (mode={depth_mode}), "
                                f"output shape: {depth_input.shape}"
                            )

                            if depth_mode == "depth_only":
                                # Depth-only mode: generate from depth structure only
                                if vace_enabled:
                                    # Use VACE conditioning with depth maps
                                    call_params["vace_input_frames"] = depth_input
                                else:
                                    # No VACE: pass depth directly as video input
                                    # The depth map becomes the "video" to transform from
                                    call_params["video"] = depth_input
                            else:
                                # V2V + Depth mode: requires VACE
                                if vace_enabled:
                                    # Pass depth to VACE, video to V2V path
                                    call_params["vace_input_frames"] = depth_input
                                    call_params["video"] = video_input
                                else:
                                    logger.warning(
                                        "V2V + Depth mode requires VACE. "
                                        "Falling back to depth-only (no VACE)."
                                    )
                                    call_params["video"] = depth_input
                        else:
                            logger.warning(
                                "Depth preprocessor enabled but model not loaded"
                            )
                            # Fall back to regular mode
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

            # === DEPTH BENCHMARK MODE ===
            # Skip main pipeline and output depth frames directly to measure depth FPS
            if DEPTH_BENCHMARK_MODE and depth_preprocessor_enabled and video_input is not None:
                # Get depth from call_params (either vace_input_frames or video depending on mode)
                # Note: can't use `or` with tensors, must check None explicitly
                depth_output = call_params.get("vace_input_frames")
                if depth_output is None:
                    depth_output = call_params.get("video")

                if depth_output is not None and isinstance(depth_output, torch.Tensor):
                    # depth_output is [1, 3, F, H, W] in [-1, 1]
                    # Convert to [F, H, W, C] in [0, 1] for output
                    depth_frames = depth_output.squeeze(0)  # [3, F, H, W]
                    depth_frames = depth_frames.permute(1, 2, 3, 0)  # [F, H, W, 3]
                    depth_frames = (depth_frames + 1.0) / 2.0  # [-1, 1] -> [0, 1]

                    output = depth_frames.float()  # [F, H, W, C] in [0, 1]

                    processing_time = time.time() - start_time
                    num_frames = output.shape[0]
                    logger.info(
                        f"[DEPTH BENCHMARK] Processed {num_frames} frames in {processing_time:.4f}s "
                        f"({num_frames / processing_time:.1f} FPS)"
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

                    for frame in output:
                        try:
                            self.output_queue.put_nowait(frame)
                        except queue.Full:
                            logger.warning("Output queue full, dropping processed frame")
                            continue

                    # Update FPS calculation
                    self._calculate_pipeline_fps(start_time, num_frames)
                    self.is_prepared = True
                    return
                else:
                    logger.warning("[DEPTH BENCHMARK] No depth output available, falling through to pipeline")

            output = pipeline(**call_params)

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
            logger.info(
                f"[Pipeline] Chunk: {num_frames} frames in {processing_time:.3f}s "
                f"({num_frames / processing_time:.1f} FPS)"
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

            for frame in output:
                try:
                    self.output_queue.put_nowait(frame)
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

    def prepare_chunk(self, chunk_size: int) -> list[torch.Tensor]:
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
            List of tensor frames, each (1, H, W, C) for downstream preprocess_chunk
        """
        # Calculate uniform sampling step
        step = len(self.frame_buffer) / chunk_size
        # Generate indices for uniform sampling
        indices = [round(i * step) for i in range(chunk_size)]
        # Extract VideoFrames at sampled indices
        video_frames = [self.frame_buffer[i] for i in indices]

        # Drop all frames up to and including the last sampled frame
        last_idx = indices[-1]
        for _ in range(last_idx + 1):
            self.frame_buffer.popleft()

        # Convert VideoFrames to tensors
        tensor_frames = []
        for video_frame in video_frames:
            # Convert VideoFrame into (1, H, W, C) tensor on cpu
            # The T=1 dimension is expected by preprocess_chunk which rearranges T H W C -> T C H W
            tensor = (
                torch.from_numpy(video_frame.to_ndarray(format="rgb24"))
                .float()
                .unsqueeze(0)
            )
            tensor_frames.append(tensor)

        return tensor_frames

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

    def _apply_depth_preprocessing(
        self,
        video_input: list[torch.Tensor],
        depth_preprocessor,
    ) -> torch.Tensor:
        """Apply Video-Depth-Anything preprocessing to video frames.

        Converts video frames to depth maps formatted for VACE conditioning.

        Args:
            video_input: List of tensor frames, each (1, H, W, C) in [0, 255]
            depth_preprocessor: VideoDepthAnything model instance

        Returns:
            Depth tensor [1, 3, F, H, W] in [-1, 1] range, ready for VACE
        """
        import torch.nn.functional as F

        # Stack frames into [F, H, W, C] tensor
        # Each frame is (1, H, W, C), so squeeze the batch dim and stack
        frames = torch.cat(video_input, dim=0)  # [F, H, W, C]

        # Run depth estimation
        # infer expects [F, H, W, C] in [0, 255] and returns [F, H, W] in [0, 1]
        depth = depth_preprocessor.infer(frames)  # [F, H, W]

        F_dim, H, W = depth.shape

        # Resize to match frame dimensions if needed
        # (depth estimation may resize internally)
        target_H, target_W = frames.shape[1], frames.shape[2]
        if H != target_H or W != target_W:
            depth = depth.unsqueeze(1)  # [F, 1, H, W]
            depth = F.interpolate(
                depth, size=(target_H, target_W), mode="bilinear", align_corners=False
            )
            depth = depth.squeeze(1)  # [F, H, W]

        # Convert single-channel to 3-channel RGB (replicate)
        depth = depth.unsqueeze(1).repeat(1, 3, 1, 1)  # [F, 3, H, W]

        # Normalize to [-1, 1] for VAE encoding
        depth = depth * 2.0 - 1.0

        # Add batch dimension and rearrange to [1, 3, F, H, W]
        depth = depth.unsqueeze(0).permute(0, 2, 1, 3, 4)

        return depth.to(dtype=torch.bfloat16)
