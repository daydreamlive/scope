import asyncio
import fractions
import logging
import threading
import time

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import AudioFrame, VideoFrame

from .frame_processor import (
    AUDIO_SAMPLES_PER_FRAME,
    WEBRTC_AUDIO_SAMPLE_RATE,
    FrameProcessor,
)
from .media_clock import MediaClock
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)


class VideoProcessingTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        fps: int = 30,
        initial_parameters: dict = None,
        notification_callback: callable = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = None
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None

        # Server-side input mode - when enabled, frames come from the backend
        # instead of WebRTC (no browser video track needed)
        self._input_source_enabled = False
        if initial_parameters:
            input_source = initial_parameters.get("input_source")
            if input_source and input_source.get("enabled"):
                self._input_source_enabled = True
                logger.info(
                    f"Input source mode enabled: {input_source.get('source_type')}"
                )

    async def input_loop(self):
        """Background loop that continuously feeds frames to the processor"""
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Stop the input loop on connection errors to avoid spam
                logger.error(f"Error in input loop, stopping: {e}")
                self.input_task_running = False
                break

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
                session_id=self.session_id,
                user_id=self.user_id,
                connection_id=self.connection_id,
                connection_info=self.connection_info,
            )
            self.frame_processor.start()

    def initialize_input_processing(self, track: MediaStreamTrack):
        self.track = track
        self.input_task_running = True
        self.input_task = asyncio.create_task(self.input_loop())

    async def recv(self) -> VideoFrame:
        """Return the next available processed frame"""
        # Lazy initialization on first call
        self.initialize_output_processing()

        # Keep running while any input source is active
        while self.input_task_running or self._input_source_enabled:
            try:
                # Update FPS: use the FPS from the pipeline chain
                if self.frame_processor:
                    self.fps = self.frame_processor.get_fps()
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
        self._input_source_enabled = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        super().stop()


class AudioProcessingTrack(MediaStreamTrack):
    """WebRTC audio track that reads from FrameProcessor's audio buffer.

    Produces 20ms audio frames (960 samples at 48kHz) synchronized with
    the video track via a shared MediaClock. When no audio data is available,
    silence frames are returned to keep the track alive.
    """

    kind = "audio"

    AUDIO_PTIME = AUDIO_SAMPLES_PER_FRAME / WEBRTC_AUDIO_SAMPLE_RATE  # 0.02s (20ms)

    def __init__(
        self,
        frame_processor: FrameProcessor,
        media_clock: MediaClock,
    ):
        super().__init__()
        self.frame_processor = frame_processor
        self.media_clock = media_clock
        self._timestamp = 0
        self._started = False
        self._last_frame_time: float | None = None

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        # Pace audio output at 20ms intervals
        if self._last_frame_time is not None:
            elapsed = time.time() - self._last_frame_time
            wait = self.AUDIO_PTIME - elapsed
            if wait > 0:
                await asyncio.sleep(wait)

        self._last_frame_time = time.time()

        # Start the shared media clock on first audio frame
        if not self._started:
            self.media_clock.start()
            self._started = True

        # Try to get audio data from the frame processor
        audio_data = self.frame_processor.get_audio(AUDIO_SAMPLES_PER_FRAME)

        if audio_data is not None:
            # Convert float32 [-1, 1] to int16 for WebRTC
            audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            # Return silence when no audio is available
            audio_int16 = np.zeros(AUDIO_SAMPLES_PER_FRAME, dtype=np.int16)

        # Create AudioFrame: shape must be (1, num_samples) for mono s16 layout
        frame = AudioFrame.from_ndarray(
            audio_int16.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = WEBRTC_AUDIO_SAMPLE_RATE

        # Set PTS from shared media clock for A/V sync
        media_time = self.media_clock.get_media_time()
        frame.pts = self.media_clock.media_time_to_audio_pts(media_time)
        frame.time_base = fractions.Fraction(1, WEBRTC_AUDIO_SAMPLE_RATE)

        return frame
