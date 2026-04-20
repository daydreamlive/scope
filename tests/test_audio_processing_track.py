"""Tests for AudioProcessingTrack's audio frame construction, resampling,
and the full recv() integration path.
"""

import asyncio
import fractions
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from av import AudioFrame

from scope.server.audio_track import (
    AUDIO_CLOCK_RATE,
    AUDIO_MAX_BUFFER_SAMPLES,
    AUDIO_PTIME,
    AudioProcessingTrack,
)
from scope.server.media_packets import AudioPacket, MediaTimestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES_PER_FRAME = int(AUDIO_CLOCK_RATE * AUDIO_PTIME)  # 960


def _make_track(channels: int = 2, started: bool = True) -> AudioProcessingTrack:
    """Create an AudioProcessingTrack with a mocked FrameProcessor.

    Args:
        channels: Number of audio channels.
        started: If True, pre-set _start so that frame helpers can be called
            without going through recv().  Set to False for recv() tests so
            the first-call initialisation branch is exercised.
    """
    fp = MagicMock()
    fp.paused = False
    fp.get_audio_packet = MagicMock(return_value=None)
    track = AudioProcessingTrack(frame_processor=fp, channels=channels)
    if started:
        track._start = time.time()
        track._timestamp = 0
    return track


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _once(audio_tensor, sample_rate: int, timestamp: MediaTimestamp | None = None):
    """Yield one AudioPacket, then None."""
    returned = False

    def _side_effect():
        nonlocal returned
        if not returned:
            returned = True
            return AudioPacket(
                audio=audio_tensor,
                sample_rate=sample_rate,
                timestamp=timestamp or MediaTimestamp(),
            )
        return None

    return _side_effect


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestStop:
    def test_clears_buffer(self):
        track = _make_track()
        track._chunks.append(np.ones(5000, dtype=np.float32))
        track._buffered_samples = 5000
        track.stop()
        assert len(track._chunks) == 0
        assert track._buffered_samples == 0


# ---------------------------------------------------------------------------
# Frame construction & edge cases
# ---------------------------------------------------------------------------


class TestFrameConstruction:
    def test_normal_samples(self):
        track = _make_track(channels=2)
        samples = np.random.uniform(-1, 1, SAMPLES_PER_FRAME * 2).astype(np.float32)
        frame = track._create_audio_frame(samples)

        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == AUDIO_CLOCK_RATE
        assert frame.samples == SAMPLES_PER_FRAME

    @pytest.mark.parametrize(
        "value, expected_int16",
        [
            (5.0, 32767),  # positive overflow clips to max
            (-5.0, -32768),  # negative overflow clips to min
            (np.inf, 32767),  # +inf clips to max
            (-np.inf, -32767),  # -inf clamped to -1.0, then scaled
        ],
    )
    def test_clipping(self, value, expected_int16):
        track = _make_track(channels=2)
        samples = np.full(SAMPLES_PER_FRAME * 2, value, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == expected_int16)

    def test_nan_becomes_silence(self):
        track = _make_track(channels=2)
        samples = np.full(SAMPLES_PER_FRAME * 2, np.nan, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_dc_offset_preserved(self):
        track = _make_track(channels=1)
        dc = 0.5
        samples = np.full(SAMPLES_PER_FRAME, dc, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == int(dc * 32767))

    def test_silence_frame_is_zeros(self):
        track = _make_track(channels=2)
        frame = track._create_silence_frame()
        assert isinstance(frame, AudioFrame)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    @pytest.mark.parametrize("channels, layout", [(1, "mono"), (2, "stereo")])
    def test_layout(self, channels, layout):
        track = _make_track(channels=channels)
        samples = np.zeros(SAMPLES_PER_FRAME * channels, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        assert frame.layout.name == layout


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    def test_same_rate_passthrough(self):
        audio = np.random.randn(2, 1000).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 48000, 48000)
        np.testing.assert_array_equal(result, audio)

    @pytest.mark.parametrize(
        "source_rate, n_in, expected_n_out",
        [
            (24000, 1000, 2000),  # 2x upsample
            (96000, 2000, 1000),  # 2x downsample
            (44100, 4410, int(round(4410 * 48000 / 44100))),  # non-integer ratio
        ],
    )
    def test_output_length(self, source_rate, n_in, expected_n_out):
        audio = np.random.randn(2, n_in).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, source_rate, 48000)
        assert result.shape == (2, expected_n_out)

    def test_single_sample(self):
        audio = np.array([[0.5], [0.5]], dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        assert result.shape[0] == 2
        assert result.shape[1] >= 1

    def test_preserves_silence(self):
        audio = np.zeros((2, 1000), dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        np.testing.assert_array_almost_equal(result, 0.0, decimal=10)

    def test_preserves_dc(self):
        dc = 0.7
        audio = np.full((1, 4800), dc, dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        # Ignore edges where interpolation tapers
        mid = result[0, 100:-100]
        np.testing.assert_allclose(mid, dc, atol=0.05)


# ---------------------------------------------------------------------------
# recv() integration (async)
# ---------------------------------------------------------------------------


class TestRecv:
    def test_no_audio_returns_silence(self):
        track = _make_track(started=False)
        frame = _run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_with_audio_tensor(self):
        track = _make_track(started=False)
        audio = torch.randn(2, SAMPLES_PER_FRAME + 100)
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000)
        )

        frame = _run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.samples == SAMPLES_PER_FRAME

    def test_resamples_to_48khz(self):
        track = _make_track(started=False)
        audio = torch.randn(2, SAMPLES_PER_FRAME)  # enough after 2x upsample
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 24000)
        )

        frame = _run(track.recv())
        assert frame.sample_rate == AUDIO_CLOCK_RATE

    def test_mono_upmixed_to_stereo(self):
        track = _make_track(started=False)
        audio = torch.randn(1, SAMPLES_PER_FRAME + 100)
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000)
        )

        frame = _run(track.recv())
        assert frame.layout.name == "stereo"

    def test_1d_tensor_treated_as_mono(self):
        track = _make_track(started=False)
        audio = torch.randn(SAMPLES_PER_FRAME + 100)  # 1D
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000)
        )

        frame = _run(track.recv())
        assert isinstance(frame, AudioFrame)

    def test_paused_returns_silence(self):
        track = _make_track(started=False)
        track.frame_processor.paused = True
        audio = torch.randn(2, SAMPLES_PER_FRAME + 100)
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000)
        )

        frame = _run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_undersized_chunk_returns_silence(self):
        track = _make_track(started=False)
        audio = torch.randn(2, 100)  # too small for a 960-sample frame
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000)
        )

        frame = _run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_accumulates_small_chunks(self):
        """Small chunks across multiple recv() calls accumulate into a real frame."""
        track = _make_track(started=False)
        chunk_size = 500  # Need 960 * 2 = 1920 interleaved → 2 chunks of 500*2=1000

        call_count = 0

        def get_audio_packet():
            nonlocal call_count
            call_count += 1
            # Return one chunk per drain cycle (odd calls), None to end drain (even)
            if call_count % 2 == 1:
                return AudioPacket(audio=torch.randn(2, chunk_size), sample_rate=48000)
            return None

        track.frame_processor.get_audio_packet = MagicMock(side_effect=get_audio_packet)

        # Call recv() repeatedly until we get a non-silence frame
        got_real_frame = False
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                frame = loop.run_until_complete(track.recv())
                raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
                if not np.all(raw == 0):
                    got_real_frame = True
                    break
        finally:
            loop.close()

        assert got_real_frame, "Should have accumulated enough for a real frame"

    def test_drains_full_queue(self):
        """A single recv() drains all available chunks, not just one."""
        track = _make_track(started=False)
        chunk_size = 200
        chunks_returned = 0

        def get_audio_packet():
            nonlocal chunks_returned
            if chunks_returned < 5:
                chunks_returned += 1
                return AudioPacket(audio=torch.randn(2, chunk_size), sample_rate=48000)
            return None

        track.frame_processor.get_audio_packet = MagicMock(side_effect=get_audio_packet)

        _run(track.recv())
        assert chunks_returned == 5
        # 5 chunks × 200 samples × 2 channels = 2000 interleaved
        # minus 960 × 2 = 1920 consumed for one frame = 80 remaining
        assert track._buffered_samples == 80

    def test_preserves_valid_packet_timestamp_at_48khz(self):
        track = _make_track(started=False)
        audio = torch.randn(2, SAMPLES_PER_FRAME + 100)
        timestamp = MediaTimestamp(pts=1234, time_base=fractions.Fraction(1, 48000))
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 48000, timestamp=timestamp)
        )

        frame = _run(track.recv())

        assert frame.pts == 1234
        assert frame.time_base == fractions.Fraction(1, 48000)

    def test_translates_preserved_timestamp_when_resampling(self):
        track = _make_track(started=False)
        input_samples = 600
        audio = torch.randn(2, input_samples)
        timestamp = MediaTimestamp(pts=2400, time_base=fractions.Fraction(1, 24000))
        track.frame_processor.get_audio_packet = MagicMock(
            side_effect=_once(audio, 24000, timestamp=timestamp)
        )

        frame = _run(track.recv())

        # 2400 @ 24kHz -> 4800 @ 48kHz
        assert frame.pts == 4800
        assert frame.time_base == fractions.Fraction(1, 48000)

    def test_caps_buffer_at_max(self):
        """Oversized buffer is trimmed to the cap after recv()."""
        track = _make_track(started=False)
        max_interleaved = AUDIO_MAX_BUFFER_SAMPLES * track.channels
        oversized = np.ones(max_interleaved + 5000, dtype=np.float32)
        track._chunks.append((oversized, None))
        track._buffered_samples = len(oversized)

        _run(track.recv())

        # After cap-trimming and consuming one 20ms frame, must be under the cap
        assert track._buffered_samples <= max_interleaved
