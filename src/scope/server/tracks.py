import asyncio
import fractions
import logging
import threading
import time
from collections import deque

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import AudioFrame, VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Audio constants
AUDIO_PTIME = 0.020  # 20ms audio frames (standard for WebRTC)
AUDIO_CLOCK_RATE = 48000  # WebRTC typically uses 48kHz for Opus codec
AUDIO_TIME_BASE = fractions.Fraction(1, AUDIO_CLOCK_RATE)


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

        # Spout input mode - when enabled, frames come from Spout instead of WebRTC
        self._spout_receiver_enabled = False
        if initial_parameters:
            spout_receiver = initial_parameters.get("spout_receiver")
            if spout_receiver and spout_receiver.get("enabled"):
                self._spout_receiver_enabled = True
                logger.info("Spout input mode enabled")

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

        # Keep running while either WebRTC input is active OR Spout input is enabled
        while self.input_task_running or self._spout_receiver_enabled:
            try:
                # Update FPS: use minimum of input FPS and pipeline FPS
                if self.frame_processor:
                    self.fps = self.frame_processor.get_output_fps()
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
        self._spout_receiver_enabled = False

        if self.input_task is not None:
            self.input_task.cancel()
            try:
                await self.input_task
            except asyncio.CancelledError:
                pass

        if self.frame_processor is not None:
            self.frame_processor.stop()

        # MediaStreamTrack.stop() is not async
        super().stop()


class AudioProcessingTrack(MediaStreamTrack):
    """WebRTC audio track that streams generated audio from the pipeline.

    This track receives audio from the FrameProcessor and streams it via WebRTC.
    Audio is resampled from the pipeline's native sample rate (typically 24kHz)
    to WebRTC's standard 48kHz for Opus codec compatibility.

    Timing follows aiortc's AudioStreamTrack pattern to ensure proper frame pacing.
    """

    kind = "audio"

    # Timing attributes (following aiortc pattern)
    _start: float
    _timestamp: int

    def __init__(
        self,
        frame_processor: FrameProcessor,
        channels: int = 2,
    ):
        """Initialize audio processing track.

        Args:
            frame_processor: The FrameProcessor instance to get audio from.
            channels: Number of audio channels (1=mono, 2=stereo). Default is stereo.
        """
        super().__init__()
        self.frame_processor = frame_processor
        self.channels = channels

        # Audio frame size: 20ms at 48kHz = 960 samples
        self._samples_per_frame = int(AUDIO_CLOCK_RATE * AUDIO_PTIME)

        # Audio buffer for accumulating samples
        # Stores resampled audio at 48kHz in interleaved format
        self._audio_buffer: deque[float] = deque()

        # Track if we've logged the first audio receipt
        self._first_audio_logged = False

        # Paused state
        self._paused = False
        self._paused_lock = threading.Lock()

        logger.info(
            f"AudioProcessingTrack initialized: channels={channels}, "
            f"output_rate={AUDIO_CLOCK_RATE}Hz, frame_size={self._samples_per_frame}"
        )

    def pause(self, paused: bool):
        """Pause or resume the audio track."""
        with self._paused_lock:
            self._paused = paused
        logger.info(f"Audio track {'paused' if paused else 'resumed'}")

    def _resample_audio(
        self, audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio from source rate to target rate.

        Args:
            audio: Audio array of shape (channels, samples)
            source_rate: Source sample rate in Hz
            target_rate: Target sample rate in Hz

        Returns:
            Resampled audio array of shape (channels, new_samples)
        """
        if source_rate == target_rate:
            return audio

        # Calculate new length
        duration = audio.shape[1] / source_rate
        new_length = int(duration * target_rate)

        # Simple linear interpolation resampling
        # For production, consider using scipy.signal.resample or librosa
        old_indices = np.linspace(0, audio.shape[1] - 1, audio.shape[1])
        new_indices = np.linspace(0, audio.shape[1] - 1, new_length)

        resampled = np.zeros((audio.shape[0], new_length), dtype=audio.dtype)
        for ch in range(audio.shape[0]):
            resampled[ch] = np.interp(new_indices, old_indices, audio[ch])

        return resampled

    async def recv(self) -> AudioFrame:
        """Return the next audio frame for WebRTC streaming.

        This method is called by aiortc to get audio frames. It:
        1. Manages timing to ensure proper frame pacing (following aiortc pattern)
        2. Fetches audio from the FrameProcessor
        3. Resamples to 48kHz if needed
        4. Buffers samples and returns 20ms frames
        """
        if self.readyState != "live":
            raise MediaStreamError

        # Timing management (following aiortc's AudioStreamTrack pattern)
        # This ensures frames are delivered at the correct rate
        if hasattr(self, "_timestamp"):
            self._timestamp += self._samples_per_frame
            wait = self._start + (self._timestamp / AUDIO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        # Check if paused
        with self._paused_lock:
            paused = self._paused

        if paused:
            return self._create_silence_frame()

        # Try to get audio from frame processor (non-blocking)
        audio_tensor, sample_rate = self.frame_processor.get_audio()

        if audio_tensor is not None and sample_rate is not None:
            logger.info(
                f"AudioTrack received audio: shape={audio_tensor.shape}, "
                f"sample_rate={sample_rate}Hz, buffer_size={len(self._audio_buffer)}"
            )
            if not self._first_audio_logged:
                self._first_audio_logged = True

            # Convert tensor to numpy: (channels, samples)
            audio_np = audio_tensor.numpy()

            # Ensure correct shape
            if audio_np.ndim == 1:
                # Mono audio, add channel dimension
                audio_np = audio_np.reshape(1, -1)

            # Resample to 48kHz if needed
            if sample_rate != AUDIO_CLOCK_RATE:
                audio_np = self._resample_audio(audio_np, sample_rate, AUDIO_CLOCK_RATE)

            # Ensure we have the right number of channels
            if audio_np.shape[0] != self.channels:
                if audio_np.shape[0] == 1 and self.channels == 2:
                    # Mono to stereo: duplicate channel
                    audio_np = np.vstack([audio_np, audio_np])
                elif audio_np.shape[0] == 2 and self.channels == 1:
                    # Stereo to mono: average channels
                    audio_np = audio_np.mean(axis=0, keepdims=True)

            # Add samples to buffer (interleaved for stereo)
            # Buffer stores samples in interleaved format: [L0, R0, L1, R1, ...]
            for i in range(audio_np.shape[1]):
                for ch in range(self.channels):
                    self._audio_buffer.append(audio_np[ch, i])

        # Check if we have enough samples for a frame
        samples_needed = self._samples_per_frame * self.channels

        if len(self._audio_buffer) >= samples_needed:
            # Extract samples for this frame
            samples = []
            for _ in range(samples_needed):
                samples.append(self._audio_buffer.popleft())

            # Create audio frame with buffered samples
            return self._create_audio_frame(samples)

        # Not enough samples, return silence
        return self._create_silence_frame()

    def _create_audio_frame(self, samples: list[float]) -> AudioFrame:
        """Create an AudioFrame from float samples.

        Args:
            samples: List of float samples in interleaved format

        Returns:
            AudioFrame ready for WebRTC transmission
        """
        # Convert float samples to int16
        # Audio from pipeline is typically in [-1, 1] range
        samples_array = np.array(samples, dtype=np.float32)
        samples_int16 = (samples_array * 32767).clip(-32768, 32767).astype(np.int16)

        # Create AudioFrame directly (following aiortc's pattern)
        layout = "stereo" if self.channels == 2 else "mono"
        frame = AudioFrame(format="s16", layout=layout, samples=self._samples_per_frame)
        frame.sample_rate = AUDIO_CLOCK_RATE
        frame.pts = self._timestamp
        frame.time_base = AUDIO_TIME_BASE

        # Update the frame's plane with the audio data
        frame.planes[0].update(samples_int16.tobytes())

        return frame

    def _create_silence_frame(self) -> AudioFrame:
        """Create a silent audio frame.

        Returns:
            AudioFrame filled with silence
        """
        # Create AudioFrame (following aiortc's AudioStreamTrack pattern)
        layout = "stereo" if self.channels == 2 else "mono"
        frame = AudioFrame(format="s16", layout=layout, samples=self._samples_per_frame)
        frame.sample_rate = AUDIO_CLOCK_RATE
        frame.pts = self._timestamp
        frame.time_base = AUDIO_TIME_BASE

        # Fill with zeros (silence)
        for p in frame.planes:
            p.update(bytes(p.buffer_size))

        return frame

    def stop(self):
        """Stop the audio track."""
        self._audio_buffer.clear()
        super().stop()
        logger.info("AudioProcessingTrack stopped")
