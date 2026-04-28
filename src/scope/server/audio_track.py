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
        self._chunks: collections.deque[tuple[np.ndarray, int | None]] = (
            collections.deque()
        )
        self._buffered_samples: int = 0  # total interleaved sample count
        self._first_audio_logged = False
        self._start: float | None = None
        self._timestamp: int = 0
        self._last_preserved_pts: int | None = None

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

        samples_needed = self._samples_per_frame * self.channels
        if self._start is None:
            await self._wait_for_initial_audio(samples_needed)
        else:
            # aiortc timing pattern: monotonic timestamp counter with wall-clock pacing
            self._timestamp += self._samples_per_frame
            wait = self._start + (self._timestamp / AUDIO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        if self.frame_processor.paused:
            return self._create_silence_frame()

        # Pull in any newly generated packets before deciding whether this
        # tick has enough buffered audio or should fall back to silence.
        self._drain_audio_packets()
        self._trim_buffer()

        # Serve a 20ms frame from the buffer
        if self._buffered_samples >= samples_needed:
            frame_parts: list[np.ndarray] = []
            remaining = samples_needed
            preserved_pts: int | None = None
            first_chunk = True

            while remaining > 0 and self._chunks:
                chunk, pts = self._chunks[0]
                take = min(len(chunk), remaining)
                if first_chunk:
                    # The emitted frame inherits the timestamp of its first
                    # output sample.
                    preserved_pts = pts
                    first_chunk = False

                frame_parts.append(chunk[:take])
                if take == len(chunk):
                    self._chunks.popleft()
                else:
                    # The leftover tail becomes a new buffered chunk whose
                    # first-sample PTS advances by the number of consumed
                    # output samples.
                    shifted_pts = None if pts is None else pts + (take // self.channels)
                    self._chunks[0] = (chunk[take:], shifted_pts)

                self._buffered_samples -= take
                remaining -= take

            frame_samples = np.concatenate(frame_parts)
            frame_pts = preserved_pts
            if frame_pts is not None:
                if (
                    self._last_preserved_pts is None
                    or frame_pts > self._last_preserved_pts
                ):
                    self._last_preserved_pts = frame_pts
                else:
                    logger.warning(
                        "Ignoring non-monotonic preserved audio timestamp in AudioProcessingTrack"
                    )
                    frame_pts = None
            return self._create_audio_frame(frame_samples, pts_override=frame_pts)

        return self._create_silence_frame()

    async def _wait_for_initial_audio(self, samples_needed: int) -> None:
        while self._buffered_samples < samples_needed:
            if self.readyState != "live":
                raise MediaStreamError

            if not self.frame_processor.paused:
                self._drain_audio_packets()
                self._trim_buffer()
                if self._buffered_samples >= samples_needed:
                    break

            await asyncio.sleep(0.01)

        self.frame_processor.notify_first_audio_packet()
        await asyncio.to_thread(self.frame_processor.wait_for_av_sync_anchor)
        self._start = time.time()
        self._timestamp = 0

    def _drain_audio_packets(self) -> None:
        # Drain all available audio from the queue to minimise latency
        # for bursty or small-chunk pipelines.
        while True:
            audio_packet = self.frame_processor.get_audio_packet()
            if audio_packet is None:
                break
            audio_tensor = audio_packet.audio
            sample_rate = audio_packet.sample_rate

            # Flush sentinel: discard buffered audio so a new prompt
            # is heard immediately instead of after the old speech finishes.
            if audio_tensor is None and sample_rate == AUDIO_FLUSH_SENTINEL:
                self._chunks.clear()
                self._buffered_samples = 0
                self._last_preserved_pts = None
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
            chunk_pts: int | None = None
            if audio_packet.timestamp.is_valid:
                # Convert the preserved timestamp onto the outgoing 48kHz sample
                # timeline so the buffered chunk stores the PTS of its first
                # output sample, regardless of the packet's original time_base.
                media_ts = audio_packet.timestamp.pts * float(
                    audio_packet.timestamp.time_base
                )
                chunk_pts = int(round(media_ts * AUDIO_CLOCK_RATE))
            self._chunks.append((interleaved, chunk_pts))
            self._buffered_samples += len(interleaved)

    def _trim_buffer(self) -> None:
        # Cap buffer to prevent unbounded growth.
        # Trim-to-tail: concatenate and keep only the newest samples so a
        # single large chunk (e.g. LTX2 delivering >1s at once) is never
        # discarded entirely.
        max_interleaved = AUDIO_MAX_BUFFER_SAMPLES * self.channels
        overflow = self._buffered_samples - max_interleaved
        while overflow > 0 and self._chunks:
            chunk, pts = self._chunks[0]
            if len(chunk) <= overflow:
                self._chunks.popleft()
                self._buffered_samples -= len(chunk)
                overflow -= len(chunk)
                continue

            # Dropping samples from the front of a buffered chunk means its
            # start PTS must move forward by the same number of output samples.
            shifted_pts = None if pts is None else pts + (overflow // self.channels)
            self._chunks[0] = (chunk[overflow:], shifted_pts)
            self._buffered_samples -= overflow
            overflow = 0

        if self._buffered_samples > max_interleaved:
            logger.warning("Audio buffer overflow, dropped oldest chunks")

    def _create_audio_frame(
        self, samples: np.ndarray, pts_override: int | None = None
    ) -> AudioFrame:
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
        frame.pts = self._timestamp if pts_override is None else pts_override
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
        self._last_preserved_pts = None
        super().stop()
