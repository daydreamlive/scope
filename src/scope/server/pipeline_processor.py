"""Pipeline processor for running a single pipeline in a thread."""

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import torch

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
DEFAULT_FPS = 30.0  # Default FPS
SLEEP_TIME = 0.01

# Input FPS measurement constants
INPUT_FPS_SAMPLE_SIZE = 30  # Number of frame intervals to track
INPUT_FPS_MIN_SAMPLES = 5  # Minimum samples needed before using input FPS


class PipelineProcessor:
    """Processes frames through a single pipeline in a dedicated thread."""

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        pipeline_id: str,
        initial_parameters: dict = None,
        track_input_fps: bool = False,
    ):
        """Initialize a pipeline processor.

        Args:
            pipeline_manager: Pipeline manager for loading pipelines
            pipeline_id: ID of the pipeline to run
            initial_parameters: Initial parameters for the pipeline
            track_input_fps: Whether to track input FPS (for first pipeline in chain)
        """
        self.pipeline_manager = pipeline_manager
        self.pipeline_id = pipeline_id
        self.track_input_fps = track_input_fps

        # Each processor creates its own queues
        self.input_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=8)

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=8)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # FPS tracking variables
        self.processing_time_per_frame = deque(maxlen=2)
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5
        self.min_fps = MIN_FPS
        self.max_fps = MAX_FPS
        self.current_pipeline_fps = DEFAULT_FPS
        self.fps_lock = threading.Lock()

        # Input FPS tracking (when enabled)
        self.input_frame_times = (
            deque(maxlen=INPUT_FPS_SAMPLE_SIZE) if track_input_fps else None
        )
        self.current_input_fps = DEFAULT_FPS if track_input_fps else None
        self.last_input_fps_update = time.time() if track_input_fps else None
        self.input_fps_lock = threading.Lock() if track_input_fps else None

        self.paused = False

        # Input mode is signaled by the frontend at stream start
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Reference to next processor in chain (if chained)
        # Used to update next processor's input_queue when output_queue is reassigned
        self.next_processor: PipelineProcessor | None = None

        # Get pipeline and check VACE status once in constructor
        self.pipeline = self.pipeline_manager.get_pipeline_by_id(self.pipeline_id)
        # Route based on frontend's VACE intent (not pipeline.vace_enabled which is lazy-loaded)
        # This fixes the chicken-and-egg problem where VACE isn't enabled until vace_input_frames arrives
        self.vace_enabled = (initial_parameters or {}).get(
            "vace_enabled", False
        )
        self.vace_use_input_video = (initial_parameters or {}).get(
            "vace_use_input_video", True
        )

    def _resize_output_queue(self, target_size: int):
        """Resize the output queue to the target size, transferring existing frames.

        Args:
            target_size: The desired maximum size for the output queue
        """
        if self.output_queue is None:
            return

        if self.output_queue.maxsize < target_size:
            logger.info(
                f"Increasing output queue size to {target_size}, current size {self.output_queue.maxsize}"
            )

            # Transfer frames from old queue to new queue
            old_queue = self.output_queue
            self.output_queue = queue.Queue(maxsize=target_size)
            while not old_queue.empty():
                try:
                    frame = old_queue.get_nowait()
                    self.output_queue.put_nowait(frame)
                except queue.Empty:
                    break

            # Update next processor's input_queue to point to the new output_queue
            if self.next_processor is not None:
                self.next_processor.input_queue = self.output_queue

    def set_next_processor(self, next_processor: "PipelineProcessor"):
        """Set the next processor in the chain and update output queue size accordingly.

        Args:
            next_processor: The next pipeline processor in the chain
        """
        self.next_processor = next_processor

        # Calculate output queue size based on next processor's requirements
        next_pipeline = next_processor.pipeline
        if hasattr(next_pipeline, "prepare"):
            requirements = next_pipeline.prepare(video=True)
            input_size = requirements.input_size
            target_size = max(8, input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR)
            self._resize_output_queue(target_size)

        # Update next processor's input_queue to point to this output_queue
        next_processor.input_queue = self.output_queue

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

        # Clear queues
        if self.input_queue:
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except queue.Empty:
                    break

        if self.output_queue:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
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

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS."""
        with self.fps_lock:
            return self.current_pipeline_fps

    def _calculate_pipeline_fps(self, start_time: float, num_frames: int):
        """Calculate FPS based on processing time and number of frames created."""
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
                    break

        logger.info(f"Worker thread stopped for pipeline: {self.pipeline_id}")

    def process_chunk(self):
        """Process a single chunk of frames."""
        start_time = time.time()

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

                # Merge new parameters with existing ones
                self.parameters = {**self.parameters, **new_parameters}
        except queue.Empty:
            pass

        # Pause or resume the processing
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            self.paused = paused
        if self.paused:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        # Prepare pipeline
        reset_cache = self.parameters.pop("reset_cache", None)
        lora_scales = self.parameters.pop("lora_scales", None)

        # Handle reset_cache: clear this processor's cache
        if reset_cache:
            logger.info(f"Clearing cache for pipeline processor: {self.pipeline_id}")
            # Clear output queue
            if self.output_queue:
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

        requirements = None
        if hasattr(self.pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected
                prepare_params["video"] = True
            requirements = self.pipeline.prepare(**prepare_params)

        video_input = None
        if requirements is not None:
            current_chunk_size = requirements.input_size

            # Check if queue has enough frames before consuming them
            if self.input_queue.qsize() < current_chunk_size:
                # Not enough frames in queue, sleep briefly and try again next iteration
                self.shutdown_event.wait(SLEEP_TIME)
                return

            video_input = []
            for _ in range(current_chunk_size):
                frame = self.input_queue.get(timeout=0.1)
                video_input.append(frame)

        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass lora_scales only when present
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Route video input based on VACE status
            # We do not support combining latent initialization and VACE conditioning
            if video_input is not None:
                # Check if pipeline actually supports VACE before routing to vace_input_frames
                from scope.core.pipelines.wan2_1.vace import VACEEnabledPipeline
                pipeline_supports_vace = isinstance(self.pipeline, VACEEnabledPipeline)

                if (
                    pipeline_supports_vace
                    and self.vace_enabled
                    and self.vace_use_input_video
                ):
                    # VACE conditioning: route to vace_input_frames
                    call_params["vace_input_frames"] = video_input
                else:
                    # Latent initialization: route to video
                    call_params["video"] = video_input

            output = self.pipeline(**call_params)

            # Clear vace_ref_images from parameters after use to prevent sending them on subsequent chunks
            # vace_ref_images should only be sent when explicitly provided in parameter updates
            if (
                "vace_ref_images" in call_params
                and "vace_ref_images" in self.parameters
            ):
                self.parameters.pop("vace_ref_images", None)

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

            processing_time = time.time() - start_time
            num_frames = output.shape[0]
            logger.debug(
                f"Pipeline {self.pipeline_id} processed in {processing_time:.4f}s, {num_frames} frames"
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
            self._resize_output_queue(target_output_queue_max_size)

            # Put frames in output queue
            # For intermediate pipelines, output goes to next pipeline's input
            # For last pipeline, output goes to frame_processor's output_queue
            # Output frames are [H, W, C], convert to [1, H, W, C] for consistency
            for frame in output:
                frame = frame.unsqueeze(0)
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    logger.debug(
                        f"Output queue full for {self.pipeline_id}, dropping processed frame"
                    )
                    # Update FPS calculation
                    self._calculate_pipeline_fps(start_time, num_frames)
                    continue

            # Update FPS calculation
            self._calculate_pipeline_fps(start_time, num_frames)
        except Exception as e:
            if self._is_recoverable(e):
                logger.error(
                    f"Error processing chunk for {self.pipeline_id}: {e}", exc_info=True
                )
            else:
                raise e

        self.is_prepared = True

    def track_input_frame(self):
        """Track timestamp of an incoming frame for FPS measurement"""
        if not self.track_input_fps:
            return

        with self.input_fps_lock:
            self.input_frame_times.append(time.time())

        # Update input FPS calculation
        self._calculate_input_fps()

    def _calculate_input_fps(self):
        """Calculate and update input FPS from recent frame timestamps."""
        if not self.track_input_fps:
            return

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

                            # Clamp to reasonable bounds
                            estimated_fps = max(
                                self.min_fps, min(self.max_fps, estimated_fps)
                            )
                            self.current_input_fps = estimated_fps

            self.last_input_fps_update = current_time

    def get_input_fps(self) -> float | None:
        """Get the current measured input FPS."""
        if not self.track_input_fps:
            return None

        with self.input_fps_lock:
            if len(self.input_frame_times) < INPUT_FPS_MIN_SAMPLES:
                return None
            return self.current_input_fps

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        return True
