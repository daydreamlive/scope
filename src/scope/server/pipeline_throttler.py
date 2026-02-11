"""Pipeline throttler for controlling frame processing rate in chained pipelines."""

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_processor import PipelineProcessor

logger = logging.getLogger(__name__)

# Throttling constants
FPS_SAMPLE_SIZE = 30
FPS_MIN_SAMPLES = 5
MIN_FPS = 1.0
MAX_FPS = 120.0

# Multiplier for target FPS when throttling
# e.g., if next pipeline processes at 6 FPS, target ~9 FPS (1.5x)
THROTTLE_TARGET_MULTIPLIER = 1.5


class PipelineThrottler:
    """Controls processing rate of a pipeline based on downstream pipeline performance.

    When pipelines are chained (A -> B -> C), a faster upstream pipeline should not
    produce frames much faster than the downstream pipeline can consume them.
    This throttler measures the downstream pipeline's input processing rate and
    adds appropriate delays to match it.

    To disable throttling, simply don't create a throttler instance (set to None).
    """

    def __init__(self):
        """Initialize the throttler."""
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Track this pipeline's output FPS (how fast it produces frames)
        self._output_times: deque[float] = deque(maxlen=FPS_SAMPLE_SIZE)
        self._output_fps: float = MAX_FPS

        # Track this pipeline's input FPS (how fast it consumes frames)
        self._input_times: deque[float] = deque(maxlen=FPS_SAMPLE_SIZE)
        self._input_fps: float = MAX_FPS

        # Reference to next processor (set externally)
        self._next_processor: PipelineProcessor | None = None

    def set_next_processor(self, processor: "PipelineProcessor | None"):
        """Set the next processor in the chain for throttling decisions.

        Args:
            processor: The next pipeline processor, or None if this is the last.
        """
        with self._lock:
            self._next_processor = processor

    def record_input_batch(self, num_frames: int, processing_time: float):
        """Record input batch processing for FPS calculation.

        Args:
            num_frames: Number of input frames in the batch.
            processing_time: Time taken to process the batch in seconds.
        """
        if num_frames <= 0 or processing_time <= 0:
            return

        with self._lock:
            current_time = time.time()
            # Record timestamps for each frame in the batch
            for i in range(num_frames):
                # Distribute timestamps across the processing time
                frame_time = (
                    current_time
                    - processing_time
                    + (processing_time * (i + 1) / num_frames)
                )
                self._input_times.append(frame_time)

            self._update_input_fps()

    def record_output_batch(self, num_frames: int, processing_time: float):
        """Record output batch for FPS calculation.

        Args:
            num_frames: Number of output frames produced.
            processing_time: Time taken to produce the batch in seconds.
        """
        if num_frames <= 0 or processing_time <= 0:
            return

        with self._lock:
            current_time = time.time()
            # Record timestamps for each frame in the batch
            for i in range(num_frames):
                frame_time = (
                    current_time
                    - processing_time
                    + (processing_time * (i + 1) / num_frames)
                )
                self._output_times.append(frame_time)

            self._update_output_fps()

    def _update_input_fps(self):
        """Update input FPS calculation. Must be called with lock held."""
        if len(self._input_times) >= FPS_MIN_SAMPLES:
            times = list(self._input_times)
            time_span = times[-1] - times[0]
            if time_span >= 0.05:  # At least 50ms
                num_frames = len(times)
                fps = num_frames / time_span
                self._input_fps = max(MIN_FPS, min(MAX_FPS, fps))

    def _update_output_fps(self):
        """Update output FPS calculation. Must be called with lock held."""
        if len(self._output_times) >= FPS_MIN_SAMPLES:
            times = list(self._output_times)
            time_span = times[-1] - times[0]
            if time_span >= 0.05:  # At least 50ms
                num_frames = len(times)
                fps = num_frames / time_span
                self._output_fps = max(MIN_FPS, min(MAX_FPS, fps))

    def get_input_fps(self) -> float:
        """Get the current input FPS (how fast this pipeline consumes frames)."""
        with self._lock:
            return self._input_fps

    def get_output_fps(self) -> float:
        """Get the current output FPS (how fast this pipeline produces frames)."""
        with self._lock:
            return self._output_fps

    def calculate_delay(self) -> float:
        """Calculate the delay needed to match downstream processing rate.

        Returns:
            Delay in seconds to sleep, or 0 if no delay needed.
        """
        next_processor = None
        output_fps = 0.0
        with self._lock:
            if self._next_processor is None:
                return 0.0
            next_processor = self._next_processor
            output_fps = self._output_fps

        next_input_fps = next_processor.throttler.get_input_fps()

        # Target FPS is slightly higher than next pipeline's input FPS
        target_fps = next_input_fps * THROTTLE_TARGET_MULTIPLIER

        # Don't throttle if we're not faster than the target
        if output_fps <= target_fps:
            return 0.0

        # Calculate delay needed per frame
        # Current interval: 1/output_fps
        # Target interval: 1/target_fps
        # Delay = target_interval - current_interval
        if target_fps <= 0:
            return 0.0

        current_interval = 1.0 / output_fps if output_fps > 0 else 0
        target_interval = 1.0 / target_fps

        delay = target_interval - current_interval

        # Only return positive delays, capped to reasonable maximum
        return max(0.0, min(delay, 1.0))

    def throttle(self):
        """Apply throttling by sleeping if necessary.

        This should be called after processing a batch and before starting the next.
        """
        delay = self.calculate_delay()
        if delay > 0:
            logger.debug(f"Throttling: sleeping {delay:.3f}s")
            self._stop_event.wait(delay)

    def interrupt(self):
        """Interrupt any in-progress throttle wait, e.g. during shutdown."""
        self._stop_event.set()
