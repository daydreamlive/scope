"""Pipeline processor for running a single pipeline in a thread."""

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import torch

from scope.core.pipelines.controller import parse_ctrl_input

from .pipeline_manager import PipelineNotAvailableException

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
    ):
        """Initialize a pipeline processor.

        Args:
            pipeline: Pipeline instance to process frames with
            pipeline_id: ID of the pipeline (used for logging)
            initial_parameters: Initial parameters for the pipeline
        """
        self.pipeline = pipeline
        self.pipeline_id = pipeline_id

        # Each processor creates its own queues
        self.input_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=8)
        # Lock to protect input_queue assignment for thread-safe reference swapping
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
        self.output_frame_times = deque(maxlen=OUTPUT_FPS_SAMPLE_SIZE)
        # Start with a higher initial FPS to prevent initial queue buildup
        self.current_output_fps = MAX_FPS
        self.output_fps_lock = threading.Lock()

        self.paused = False
        # Input mode is signaled by the frontend at stream start
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Reference to next processor in chain (if chained)
        # Used to update next processor's input_queue when output_queue is reassigned
        self.next_processor: PipelineProcessor | None = None

        # Route based on frontend's VACE intent (not pipeline.vace_enabled which is lazy-loaded)
        # This fixes the chicken-and-egg problem where VACE isn't enabled until vace_input_frames arrives
        self.vace_enabled = (initial_parameters or {}).get("vace_enabled", False)
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
            # Use lock to ensure thread-safe reference swapping
            if self.next_processor is not None:
                with self.next_processor.input_queue_lock:
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
        # Use lock to ensure thread-safe reference swapping
        with next_processor.input_queue_lock:
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
        with self.input_queue_lock:
            input_queue_ref = self.input_queue
        if input_queue_ref:
            while not input_queue_ref.empty():
                try:
                    input_queue_ref.get_nowait()
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

    def prepare_chunk(
        self, input_queue_ref: queue.Queue, chunk_size: int
    ) -> list[torch.Tensor]:
        """
        Sample frames uniformly from the queue, convert them to tensors, and remove processed frames.

        This function implements uniform sampling across the entire queue to ensure
        temporal coverage of input frames. It samples frames at evenly distributed
        indices and removes all frames up to the last sampled frame to prevent
        queue buildup.

        Note:
            This function must be called with a queue reference obtained while holding
            input_queue_lock. The caller is responsible for thread safety.

        Example:
            With queue_size=8 and chunk_size=4:
            - step = 8/4 = 2.0
            - indices = [0, 2, 4, 6] (uniformly distributed)
            - Returns frames at positions 0, 2, 4, 6
            - Removes frames 0-6 from queue (7 frames total)
            - Keeps frame 7 in queue

        Args:
            input_queue_ref: Reference to the input queue (obtained while holding lock)
            chunk_size: Number of frames to sample

        Returns:
            List of tensor frames, each (1, H, W, C) for downstream preprocess_chunk
        """

        # Calculate uniform sampling step
        step = input_queue_ref.qsize() / chunk_size
        # Generate indices for uniform sampling
        indices = [round(i * step) for i in range(chunk_size)]
        # Extract VideoFrames at sampled indices
        video_frames = []

        # Drop all frames up to and including the last sampled frame
        last_idx = indices[-1]
        for i in range(last_idx + 1):
            frame = input_queue_ref.get_nowait()
            if i in indices:
                video_frames.append(frame)
        return video_frames

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

            # Capture a local reference to input_queue while holding the lock
            # This ensures thread-safe access even if input_queue is reassigned
            with self.input_queue_lock:
                input_queue_ref = self.input_queue

            # Check if queue has enough frames before consuming them
            if input_queue_ref.qsize() < current_chunk_size:
                # Not enough frames in queue, sleep briefly and try again next iteration
                self.shutdown_event.wait(SLEEP_TIME)
                return

            # Use prepare_chunk to uniformly sample frames from the queue
            video_input = self.prepare_chunk(input_queue_ref, current_chunk_size)

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

            # Extract ctrl_input, parse it, and reset mouse for next frame
            if "ctrl_input" in self.parameters:
                ctrl_data = self.parameters["ctrl_input"]
                call_params["ctrl_input"] = parse_ctrl_input(ctrl_data)
                # Reset mouse accumulator, keep key state
                self.parameters["ctrl_input"]["mouse"] = [0.0, 0.0]

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
                    # Track when a frame is added to output queue for FPS calculation
                    self._track_output_frame()
                except queue.Full:
                    logger.info(
                        f"Output queue full for {self.pipeline_id}, dropping processed frame"
                    )
                    continue
        except Exception as e:
            if self._is_recoverable(e):
                logger.error(
                    f"Error processing chunk for {self.pipeline_id}: {e}", exc_info=True
                )
            else:
                raise e

        self.is_prepared = True

    def _track_output_frame(self):
        """Track when a frame is added to the output queue (production rate)."""
        with self.output_fps_lock:
            self.output_frame_times.append(time.time())

        self._calculate_output_fps()

    def _calculate_output_fps(self):
        """Calculate FPS based on how fast frames are produced into the output queue."""
        with self.output_fps_lock:
            if len(self.output_frame_times) >= OUTPUT_FPS_MIN_SAMPLES:
                times = list(self.output_frame_times)
                # Time span from first to last frame
                time_span = times[-1] - times[0]
                # Require minimum time span to avoid unstable estimates from very short intervals
                if time_span >= 0.05:  # At least 50ms
                    # FPS = number of frames / time_span
                    # e.g., if 4 frames are produced in 1s, then we have 4 FPS
                    num_frames = len(times)
                    estimated_fps = num_frames / time_span

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
