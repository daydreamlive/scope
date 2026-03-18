import asyncio
import collections
import fractions
import logging
import threading
import time

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import VIDEO_CLOCK_RATE, VIDEO_TIME_BASE, MediaStreamError
from av import AudioFrame, VideoFrame

from .frame_processor import FrameProcessor
from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Audio constants
AUDIO_CLOCK_RATE = 48000  # Standard WebRTC audio clock rate (48 kHz for Opus)
AUDIO_PTIME = 0.020  # 20ms audio frames (standard for WebRTC)
AUDIO_TIME_BASE = fractions.Fraction(1, AUDIO_CLOCK_RATE)
# Maximum buffered audio before we start dropping oldest samples (1 second)
AUDIO_MAX_BUFFER_SAMPLES = AUDIO_CLOCK_RATE


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
        tempo_sync=None,
        frame_processor: "FrameProcessor | None" = None,
    ):
        super().__init__()
        self.pipeline_manager = pipeline_manager
        self.initial_parameters = initial_parameters or {}
        self.notification_callback = notification_callback
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info
        self.tempo_sync = tempo_sync
        # FPS variables (will be updated from FrameProcessor or input measurement)
        self.fps = fps
        self.frame_ptime = 1.0 / fps

        self.frame_processor = frame_processor
        self.input_task = None
        self.input_task_running = False
        self._paused = False
        self._paused_lock = threading.Lock()
        self._last_frame = None
        self._last_send_time: float | None = None
        self._pts: int = 0
        self._frame_lock = threading.Lock()

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
        consecutive_errors = 0
        max_consecutive_errors = 10
        while self.input_task_running:
            try:
                input_frame = await self.track.recv()
                consecutive_errors = 0

                # Store raw VideoFrame for later processing (tracks input FPS internally)
                self.frame_processor.put(input_frame)

            except asyncio.CancelledError:
                break
            except MediaStreamError:
                logger.info("Source track ended")
                self.input_task_running = False
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        f"Error in input loop, stopping after "
                        f"{consecutive_errors} consecutive errors: {e}"
                    )
                    self.input_task_running = False
                    break
                logger.warning(
                    f"Transient error in input loop "
                    f"({consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                await asyncio.sleep(0.01)

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        """Pace output at the target frame rate and return a monotonic PTS."""
        if self.readyState != "live":
            raise MediaStreamError

        # Pace frames at the target interval
        if self._last_send_time is not None:
            elapsed = time.time() - self._last_send_time
            wait = self.frame_ptime - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._pts += int(self.frame_ptime * VIDEO_CLOCK_RATE)

        self._last_send_time = time.time()

        return self._pts, VIDEO_TIME_BASE

    def initialize_output_processing(self):
        """No-op guard; FrameProcessor is injected via constructor."""
        if not self.frame_processor:
            raise RuntimeError(
                "VideoProcessingTrack requires a FrameProcessor. "
                "Pass one via the constructor."
            )

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

                    with self._frame_lock:
                        self._last_frame = frame
                    return frame

                # No frame available, wait a bit before trying again
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Error getting processed frame: {e}")
                raise

        raise Exception("Track stopped")

    def get_last_frame(self):
        """Return the most recently rendered frame, or None."""
        with self._frame_lock:
            return self._last_frame

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

        # Note: frame_processor.stop() is handled by Session.close(),
        # not here, because the FrameProcessor is shared with AudioProcessingTrack.

        super().stop()


class AudioProcessingTrack(MediaStreamTrack):
    """WebRTC audio track that streams generated audio from the pipeline.

    Receives raw audio chunks from FrameProcessor, resamples to 48kHz
    via linear interpolation, buffers samples, and delivers 20ms stereo
    frames for WebRTC/Opus.

    Timing follows aiortc's AudioStreamTrack pattern (monotonic _timestamp
    counter) to ensure proper frame pacing without wall-clock drift.
    """

    kind = "audio"

    def __init__(
        self,
        frame_processor: FrameProcessor,
        channels: int = 2,
    ):
        super().__init__()
        self.frame_processor = frame_processor
        self.channels = channels

        self._samples_per_frame = int(AUDIO_CLOCK_RATE * AUDIO_PTIME)  # 960
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._buffered_samples: int = 0  # total interleaved sample count
        self._first_audio_logged = False
        self._start: float | None = None
        self._timestamp: int = 0

    @staticmethod
    def _resample_audio(
        audio: np.ndarray, source_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio (channels, samples) using linear interpolation.

        Linear interpolation is chunk-boundary safe (no spectral leakage
        artifacts from treating each chunk as periodic) and fast enough for
        real-time 20ms frame delivery.
        """
        if source_rate == target_rate:
            return audio

        n_in = audio.shape[1]
        n_out = int(round(n_in * target_rate / source_rate))
        if n_out == n_in:
            return audio

        x_in = np.linspace(0, 1, n_in, dtype=np.float64)
        x_out = np.linspace(0, 1, n_out, dtype=np.float64)

        resampled = np.empty((audio.shape[0], n_out), dtype=np.float32)
        for ch in range(audio.shape[0]):
            resampled[ch] = np.interp(x_out, x_in, audio[ch])
        return resampled

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError

        # aiortc timing pattern: monotonic timestamp counter with wall-clock pacing
        if self._start is not None:
            self._timestamp += self._samples_per_frame
            wait = self._start + (self._timestamp / AUDIO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0

        if self.frame_processor.paused:
            return self._create_silence_frame()

        # Drain all available audio from the queue to minimise latency
        # for bursty or small-chunk pipelines.
        while True:
            audio_tensor, sample_rate = self.frame_processor.get_audio()
            if audio_tensor is None or sample_rate is None:
                break

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

            # Interleave channels into packed format for PyAV's "s16" layout.
            # audio_np is (channels, samples). Fortran-order ravel traverses
            # columns first, producing [L0, R0, L1, R1, ...] which is exactly
            # what packed interleaved s16 expects in a single plane.
            interleaved = np.ravel(audio_np, order="F").astype(np.float32)
            self._chunks.append(interleaved)
            self._buffered_samples += len(interleaved)

        # Cap buffer to prevent unbounded growth (1 second of interleaved audio)
        max_interleaved = AUDIO_MAX_BUFFER_SAMPLES * self.channels
        if self._buffered_samples > max_interleaved:
            while self._buffered_samples > max_interleaved and self._chunks:
                dropped = self._chunks.popleft()
                self._buffered_samples -= len(dropped)
            logger.warning("Audio buffer overflow, dropped oldest chunks")

        # Serve a 20ms frame from the buffer
        samples_needed = self._samples_per_frame * self.channels
        if self._buffered_samples >= samples_needed:
            flat = np.concatenate(list(self._chunks))
            self._chunks.clear()
            frame_samples = flat[:samples_needed]
            remainder = flat[samples_needed:]
            if len(remainder) > 0:
                self._chunks.append(remainder)
            self._buffered_samples = len(remainder)
            return self._create_audio_frame(frame_samples)

        return self._create_silence_frame()

    def _create_audio_frame(self, samples: np.ndarray) -> AudioFrame:
        """Build a packed s16 AudioFrame from interleaved float32 samples.

        ``samples`` must already be interleaved: [L0, R0, L1, R1, …] for
        stereo.  PyAV's ``s16`` (not ``s16p``) stores all channels in a
        single plane in packed order, so we write directly to ``planes[0]``.
        """
        samples_int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)

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
        frame.pts = self._timestamp
        frame.time_base = AUDIO_TIME_BASE
        for p in frame.planes:
            p.update(bytes(p.buffer_size))
        return frame

    def stop(self):
        self._chunks.clear()
        self._buffered_samples = 0
        super().stop()
