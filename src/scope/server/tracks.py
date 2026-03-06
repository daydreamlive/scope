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
from .media_clock import MediaClock
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
        media_clock: MediaClock | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.media_clock = media_clock
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
        self._last_send_time: float | None = None
        self._clock_started = False

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

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Pace output at the target frame rate and return a PTS from the shared MediaClock.

        Using the shared clock ensures the video PTS is correlated with the
        audio PTS so the WebRTC receiver can synchronize playback.
        """
        if self.readyState != "live":
            raise MediaStreamError

        # Pace frames at the target interval
        if self._last_send_time is not None:
            elapsed = time.time() - self._last_send_time
            wait = self.frame_ptime - elapsed
            if wait > 0:
                await asyncio.sleep(wait)

        self._last_send_time = time.time()

        # Start the shared clock on the first frame (idempotent)
        if self.media_clock and not self._clock_started:
            self.media_clock.start()
            self._clock_started = True

        if self.media_clock:
            return self.media_clock.to_pts(VIDEO_CLOCK_RATE), VIDEO_TIME_BASE

        # Fallback for cases without a media clock (shouldn't happen in normal flow)
        if not hasattr(self, "_fallback_pts"):
            self._fallback_pts = 0
        else:
            self._fallback_pts += int(self.frame_ptime * VIDEO_CLOCK_RATE)
        return self._fallback_pts, VIDEO_TIME_BASE

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

        # Propagate to frame_processor so AudioProcessingTrack can check it
        if self.frame_processor:
            self.frame_processor.paused = paused

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
    """WebRTC audio track that streams generated audio from the pipeline.

    Receives raw audio chunks from FrameProcessor, resamples to 48kHz,
    buffers samples, and delivers 20ms stereo frames for WebRTC/Opus.

    Timing follows aiortc's AudioStreamTrack pattern (monotonic _timestamp
    counter) to ensure proper frame pacing without wall-clock drift.
    """

    kind = "audio"

    _start: float
    _timestamp: int

    def __init__(
        self,
        frame_processor: FrameProcessor,
        channels: int = 2,
    ):
        super().__init__()
        self.frame_processor = frame_processor
        self.channels = channels

        self._samples_per_frame = int(AUDIO_CLOCK_RATE * AUDIO_PTIME)  # 960
        self._audio_buffer: deque[float] = deque()
        self._first_audio_logged = False

    @staticmethod
    def _resample_audio(
        audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio (channels, samples) via FFT for clean upsampling."""
        if source_rate == target_rate:
            return audio

        n_in = audio.shape[1]
        n_out = int(round(n_in * target_rate / source_rate))
        if n_out == n_in:
            return audio

        resampled = np.zeros((audio.shape[0], n_out), dtype=np.float32)
        for ch in range(audio.shape[0]):
            spectrum = np.fft.rfft(audio[ch])
            n_freq_out = n_out // 2 + 1
            if n_freq_out > len(spectrum):
                padded = np.zeros(n_freq_out, dtype=spectrum.dtype)
                padded[: len(spectrum)] = spectrum
                resampled[ch] = np.fft.irfft(padded, n=n_out) * (n_out / n_in)
            else:
                resampled[ch] = np.fft.irfft(spectrum[:n_freq_out], n=n_out) * (
                    n_out / n_in
                )
        return resampled

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        # aiortc timing pattern: monotonic timestamp counter with wall-clock pacing
        if hasattr(self, "_timestamp"):
            self._timestamp += self._samples_per_frame
            wait = self._start + (self._timestamp / AUDIO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        if self.frame_processor.paused:
            return self._create_silence_frame()

        # Pull audio from the pipeline (non-blocking)
        audio_tensor, sample_rate = self.frame_processor.get_audio()

        if audio_tensor is not None and sample_rate is not None:
            if not self._first_audio_logged:
                logger.info(
                    f"AudioTrack received first audio: shape={audio_tensor.shape}, "
                    f"sample_rate={sample_rate}Hz"
                )
                self._first_audio_logged = True

            audio_np = audio_tensor.numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np.reshape(1, -1)

            # Resample to 48kHz
            if sample_rate != AUDIO_CLOCK_RATE:
                audio_np = self._resample_audio(audio_np, sample_rate, AUDIO_CLOCK_RATE)

            # Channel conversion
            if audio_np.shape[0] != self.channels:
                if audio_np.shape[0] == 1 and self.channels == 2:
                    audio_np = np.vstack([audio_np, audio_np])
                elif audio_np.shape[0] == 2 and self.channels == 1:
                    audio_np = audio_np.mean(axis=0, keepdims=True)

            # Interleave into buffer: [L0, R0, L1, R1, ...]
            for i in range(audio_np.shape[1]):
                for ch in range(self.channels):
                    self._audio_buffer.append(audio_np[ch, i])

        # Serve a 20ms frame from the buffer
        samples_needed = self._samples_per_frame * self.channels
        if len(self._audio_buffer) >= samples_needed:
            samples = [self._audio_buffer.popleft() for _ in range(samples_needed)]
            return self._create_audio_frame(samples)

        return self._create_silence_frame()

    def _create_audio_frame(self, samples: list[float]) -> AudioFrame:
        samples_array = np.array(samples, dtype=np.float32)
        samples_int16 = (samples_array * 32767).clip(-32768, 32767).astype(np.int16)

        layout = "stereo" if self.channels == 2 else "mono"
        frame = AudioFrame(format="s16", layout=layout, samples=self._samples_per_frame)
        frame.sample_rate = AUDIO_CLOCK_RATE
        frame.pts = self._timestamp
        frame.time_base = AUDIO_TIME_BASE
        frame.planes[0].update(samples_int16.tobytes())
        return frame

    def _create_silence_frame(self) -> AudioFrame:
        layout = "stereo" if self.channels == 2 else "mono"
        frame = AudioFrame(format="s16", layout=layout, samples=self._samples_per_frame)
        frame.sample_rate = AUDIO_CLOCK_RATE
        frame.pts = getattr(self, "_timestamp", 0)
        frame.time_base = AUDIO_TIME_BASE
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        return frame

    def stop(self):
        self._audio_buffer.clear()
        super().stop()
