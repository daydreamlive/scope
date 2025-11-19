import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import torch
from aiortc.mediastreams import VideoFrame

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
DEFAULT_FPS = 30.0  # Default FPS
SLEEP_TIME = 0.01


class FrameProcessor:
    def __init__(
        self,
        pipeline_manager: PipelineManager,
        max_output_queue_size: int = 8,
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

        self.paused = False

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()
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

        with self.frame_buffer_lock:
            self.frame_buffer.append(frame)
            return True

    def get(self) -> torch.Tensor | None:
        if not self.running:
            return None

        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS"""
        with self.fps_lock:
            return self.current_pipeline_fps

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

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        # Put new parameters in queue (replace any pending update)
        try:
            # Add new update
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.info("Parameter queue full, dropping parameter update")
            return False

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
            requirements = pipeline.prepare(
                **self.parameters,
            )

        video_input = None
        if requirements is not None:
            current_chunk_size = requirements.input_size
            with self.frame_buffer_lock:
                if not self.frame_buffer or len(self.frame_buffer) < current_chunk_size:
                    # Sleep briefly to avoid busy waiting
                    self.shutdown_event.wait(SLEEP_TIME)
                    return
                video_input = self.prepare_chunk(current_chunk_size)
        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass video input to pipeline
            if video_input is not None:
                call_params["video"] = video_input

            output = pipeline(**call_params)

            # Clear transition when complete (blocks signal completion via _transition_active)
            # Contract: Modular pipelines manage prompts internally; frame_processor manages lifecycle
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(pipeline, "state"):
                    transition_active = pipeline.state.get("_transition_active", False)

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    self.parameters.pop("transition", None)

            processing_time = time.time() - start_time
            num_frames = output.shape[0]
            logger.debug(
                f"Processed pipeline in {processing_time:.4f}s, {num_frames} frames"
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
            List of processed tensor frames, or empty list if insufficient buffer
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
            # Convert VideoFrame into THWC tensor on cpu
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
