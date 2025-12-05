import asyncio
import fractions
import logging
import threading
import time
from collections import deque

from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Input FPS measurement constants
INPUT_FPS_SAMPLE_SIZE = 30  # Number of frame intervals to track
INPUT_FPS_MIN_SAMPLES = 5  # Minimum samples needed before using input FPS


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None

        # Input FPS measurement: track timestamps of incoming frames
        self._input_frame_times = deque(maxlen=INPUT_FPS_SAMPLE_SIZE)
        self._input_fps_lock = threading.Lock()

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Track input frame timestamp for FPS measurement
                with self._input_fps_lock:
                    self._input_frame_times.append(time.time())

                # Store raw VideoFrame for later processing
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

    def _get_input_fps(self) -> float | None:
        """Calculate input FPS from recent frame timestamps.

        Returns the measured input FPS if enough samples are available,
        otherwise returns None to indicate fallback should be used.
        """
        with self._input_fps_lock:
            if len(self._input_frame_times) < INPUT_FPS_MIN_SAMPLES:
                return None

            # Calculate FPS from frame intervals
            times = list(self._input_frame_times)
            if len(times) < 2:
                return None

            # Time span from first to last frame
            time_span = times[-1] - times[0]
            if time_span <= 0:
                return None

            # FPS = (number of intervals) / time_span
            num_intervals = len(times) - 1
            return num_intervals / time_span

    # Copied from https://github.com/livepeer/fastworld/blob/e649ef788cd33d78af6d8e1da915cd933761535e/backend/track.py#L267
    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Override to control frame rate"""
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "timestamp"):
            # Calculate wait time based on current frame rate
            current_time = time.time()
            time_since_last_frame = current_time - self.last_frame_time

            # Wait for the appropriate interval based on current FPS
            target_interval = self.frame_ptime  # Current frame period
            wait_time = target_interval - time_since_last_frame

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Update timestamp and last frame time
            self.timestamp += int(self.frame_ptime * VIDEO_CLOCK_RATE)
            self.last_frame_time = time.time()
        else:
            self.start = time.time()
            self.last_frame_time = time.time()
            self.timestamp = 0

        return self.timestamp, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        if not self.frame_processor:
            self.frame_processor = FrameProcessor(
                pipeline_manager=self.pipeline_manager,
                initial_parameters=self.initial_parameters,
                notification_callback=self.notification_callback,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        # Clear input frame times for fresh FPS measurement
        with self._input_fps_lock:
            self._input_frame_times.clear()
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()
        while self.input_task_running:
            try:
                # Update FPS: prefer input FPS measurement, fall back to FrameProcessor
                input_fps = self._get_input_fps()
                if input_fps is not None:
                    self.fps = input_fps
                elif self.frame_processor:
                    self.fps = self.frame_processor.get_current_pipeline_fps()
                self.frame_ptime = 1.0 / self.fps

                # If paused, wait for the appropriate frame interval before returning
                with self._paused_lock:
                    paused = self._paused

                frame = None
                if paused:
                    # When video is paused, return the last frame to freeze the playback video
                    frame = self._last_frame
                else:
                    # When video is not paused, get the next frame from the frame processor
                    frame_tensor = self.frame_processor.get()
                    if frame_tensor is not None:
                        frame = VideoFrame.from_ndarray(
                            frame_tensor.numpy(), format="rgb24"
                        )

                if frame is not None:
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    self._last_frame = frame
                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def pause(self, paused: bool):
        """Pause or resume the video track processing"""
        with self._paused_lock:
            self._paused = paused
        logger.info(f"Video track {'paused' if paused else 'resumed'}")

    async def stop(self):
        self.input_task_running = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        # Clear input frame times
        with self._input_fps_lock:
            self._input_frame_times.clear()

        await super().stop()
