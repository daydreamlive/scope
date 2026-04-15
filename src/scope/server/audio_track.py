import asyncio
import collections
import fractions
import logging
import time

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from .frame_processor import FrameProcessor
from .pipeline_processor import AUDIO_FLUSH_SENTINEL

logger = logging.getLogger(__name__)

# Audio constants
AUDIO_CLOCK_RATE = 48000  # Standard WebRTC audio clock rate (48 kHz for Opus)
AUDIO_PTIME = 0.020  # 20ms audio frames (standard for WebRTC)
AUDIO_TIME_BASE = fractions.Fraction(1, AUDIO_CLOCK_RATE)
# Maximum buffered audio before we start dropping oldest samples (60 seconds).
# Must be large enough for bursty pipelines like LTX2 that deliver >1s of audio
# in a single chunk after each denoising pass, and for TTS pipelines that may
# generate many seconds of speech before playback catches up.
AUDIO_MAX_BUFFER_SAMPLES = AUDIO_CLOCK_RATE * 60


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

        samples_needed = self._samples_per_frame * self.channels

        # Lazy drain: pull chunks from the queue only until the local buffer
        # has enough samples to serve this frame. Leaving unconsumed chunks
        # in ``audio_output_queue`` creates natural backpressure — producers
        # block on the (small) queue's blocking put and cascade the stall
        # upstream through node-to-node edge queues, matching production
        # rate to real-time consumption without silent drops.
        while self._buffered_samples < samples_needed:
            audio_tensor, sample_rate = self.frame_processor.get_audio()
            if audio_tensor is None and sample_rate is None:
                break

            # Flush sentinel: discard buffered audio so a new prompt
            # is heard immediately instead of after the old speech finishes.
            if audio_tensor is None and sample_rate == AUDIO_FLUSH_SENTINEL:
                self._chunks.clear()
                self._buffered_samples = 0
                logger.info("Audio buffer flushed (prompt change)")
                continue

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

        # Safety net: if something managed to push more than the cap anyway
        # (shouldn't happen with lazy drain + maxsize=1 backpressure), keep
        # the tail so we don't grow unbounded.
        max_interleaved = AUDIO_MAX_BUFFER_SAMPLES * self.channels
        if self._buffered_samples > max_interleaved:
            flat = np.concatenate(list(self._chunks))
            self._chunks.clear()
            trimmed = flat[-max_interleaved:]
            self._chunks.append(trimmed)
            self._buffered_samples = len(trimmed)
            logger.warning("Audio buffer overflow, dropped oldest chunks")

        # Serve a 20ms frame from the buffer
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
        samples_int16 = (
            (np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0) * 32767)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

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
