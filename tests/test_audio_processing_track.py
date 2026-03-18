"""Tests for AudioProcessingTrack's audio frame construction, resampling,
and the full recv() integration path.
"""

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import torch
from av import AudioFrame

from scope.server.tracks import (
    AUDIO_CLOCK_RATE,
    AUDIO_PTIME,
    AudioProcessingTrack,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_track(channels: int = 2, init_timestamp: bool = True) -> AudioProcessingTrack:
    """Create an AudioProcessingTrack with a mocked FrameProcessor.

    Args:
        channels: Number of audio channels.
        init_timestamp: If True, pre-set _start and _timestamp so that
            _create_audio_frame can be called without going through recv().
            Set to False for tests that exercise recv() directly.
    """
    fp = MagicMock()
    fp.paused = False
    fp.get_audio = MagicMock(return_value=(None, None))
    track = AudioProcessingTrack(frame_processor=fp, channels=channels)
    if init_timestamp:
        track._start = time.time()
        track._timestamp = 0
    return track


SAMPLES_PER_FRAME = int(AUDIO_CLOCK_RATE * AUDIO_PTIME)  # 960


def _once(audio_tensor, sample_rate: int):
    """Return a side_effect callable that yields (tensor, rate) once, then (None, None)."""
    returned = False

    def _side_effect():
        nonlocal returned
        if not returned:
            returned = True
            return (audio_tensor, sample_rate)
        return (None, None)

    return _side_effect


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


class TestStop:
    """Test stop() side-effects."""

    def test_empty_buffer_after_stop(self):
        """stop() should clear the buffer."""
        track = _make_track(channels=2)
        track._audio_buffer = np.ones(5000, dtype=np.float32)
        track.stop()
        assert len(track._audio_buffer) == 0


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


class TestFrameConstruction:
    """Test _create_audio_frame and _create_silence_frame."""

    def test_create_audio_frame_normal(self):
        """Normal float32 samples should produce a valid s16 AudioFrame."""
        track = _make_track(channels=2)
        samples = np.random.uniform(-1, 1, SAMPLES_PER_FRAME * 2).astype(np.float32)
        frame = track._create_audio_frame(samples)

        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == AUDIO_CLOCK_RATE
        assert frame.samples == SAMPLES_PER_FRAME

    def test_create_audio_frame_clipping(self):
        """Values outside [-1, 1] must be clipped, not wrap around."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, 5.0, dtype=np.float32)  # way above 1.0
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 32767), "Positive overflow should clip to 32767"

    def test_create_audio_frame_negative_clipping(self):
        """Large negative values must clip to -32768."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, -5.0, dtype=np.float32)
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == -32768)

    def test_create_silence_frame(self):
        """Silence frame should be all zeros."""
        track = _make_track(channels=2)
        frame = track._create_silence_frame()

        assert isinstance(frame, AudioFrame)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_mono_frame_layout(self):
        """Mono track should produce a mono-layout AudioFrame."""
        track = _make_track(channels=1)
        samples = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        assert frame.layout.name == "mono"

    def test_stereo_frame_layout(self):
        """Stereo track should produce a stereo-layout AudioFrame."""
        track = _make_track(channels=2)
        samples = np.zeros(SAMPLES_PER_FRAME * 2, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        assert frame.layout.name == "stereo"


# ---------------------------------------------------------------------------
# Adversarial frame inputs
# ---------------------------------------------------------------------------


class TestAdversarialInputs:
    """Edge-case inputs for _create_audio_frame."""

    def test_nan_values_dont_crash(self):
        """NaN in audio should not crash frame creation."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, np.nan, dtype=np.float32)
        frame = track._create_audio_frame(samples)
        assert isinstance(frame, AudioFrame)

    def test_inf_values_get_clipped(self):
        """Inf values should clip to int16 bounds."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, np.inf, dtype=np.float32)
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 32767)

    def test_negative_inf_values_get_clipped(self):
        """-Inf values should clip to -32768."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, -np.inf, dtype=np.float32)
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == -32768)

    def test_dc_offset_preserved(self):
        """A constant DC offset should survive frame creation intact."""
        track = _make_track(channels=1)
        dc = 0.5
        n = SAMPLES_PER_FRAME
        samples = np.full(n, dc, dtype=np.float32)
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        expected = int(dc * 32767)
        assert np.all(raw == expected)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    """Test the FFT-based resampler."""

    def test_same_rate_passthrough(self):
        """Same source and target rate should return the input unchanged."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 48000, 48000)
        np.testing.assert_array_equal(result, audio)

    def test_upsample_length(self):
        """Upsampling 24kHz -> 48kHz should double the sample count."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        assert result.shape == (2, 2000)

    def test_downsample_length(self):
        """Downsampling 96kHz -> 48kHz should halve the sample count."""
        audio = np.random.randn(2, 2000).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 96000, 48000)
        assert result.shape == (2, 1000)

    def test_resample_single_sample(self):
        """Resampling a single sample should not crash."""
        audio = np.array([[0.5], [0.5]], dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        assert result.shape[0] == 2
        assert result.shape[1] >= 1

    def test_resample_preserves_silence(self):
        """All-zero input should remain all-zero after resampling."""
        audio = np.zeros((2, 1000), dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        np.testing.assert_array_almost_equal(result, 0.0, decimal=10)

    def test_resample_preserves_dc(self):
        """A DC signal should roughly preserve its level after resampling."""
        dc = 0.7
        audio = np.full((1, 4800), dc, dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        mid = result[0, 100:-100]
        np.testing.assert_allclose(mid, dc, atol=0.05)

    def test_resample_odd_ratio(self):
        """Non-integer ratio (44100 -> 48000) should not crash."""
        audio = np.random.randn(2, 4410).astype(np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 44100, 48000)
        expected_len = int(round(4410 * 48000 / 44100))
        assert result.shape == (2, expected_len)

    def test_resample_very_short_audio(self):
        """Two-sample audio resampled should not crash."""
        audio = np.array([[0.1, -0.1], [0.2, -0.2]], dtype=np.float32)
        result = AudioProcessingTrack._resample_audio(audio, 24000, 48000)
        assert result.shape[0] == 2
        assert result.shape[1] >= 2


# ---------------------------------------------------------------------------
# Full recv() integration (async)
# ---------------------------------------------------------------------------


class TestRecvIntegration:
    """Integration tests that exercise the full recv() path."""

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_recv_no_audio_returns_silence(self):
        """When no audio is queued, recv() should return a silence frame."""
        track = _make_track(channels=2, init_timestamp=False)
        frame = self._run(track.recv())

        assert isinstance(frame, AudioFrame)
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_recv_with_audio_tensor(self):
        """recv() should process a torch tensor and return an AudioFrame."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(2, n_samples)
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(audio_tensor, 48000)
        )

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.samples == SAMPLES_PER_FRAME

    def test_recv_with_resampling(self):
        """recv() should resample 24kHz audio to 48kHz."""
        track = _make_track(channels=2, init_timestamp=False)
        n_input = SAMPLES_PER_FRAME  # more than enough after upsampling
        audio_tensor = torch.randn(2, n_input)
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(audio_tensor, 24000)
        )

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == AUDIO_CLOCK_RATE

    def test_recv_mono_input_stereo_output(self):
        """Mono audio should be upmixed to stereo in recv()."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(1, n_samples)  # mono
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(audio_tensor, 48000)
        )

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.layout.name == "stereo"

    def test_recv_1d_tensor(self):
        """A 1D tensor (no channel dim) should be handled as mono."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(n_samples)  # 1D
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(audio_tensor, 48000)
        )

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)

    def test_recv_paused_returns_silence(self):
        """When paused, recv() should return silence regardless of queued audio."""
        track = _make_track(channels=2, init_timestamp=False)
        track.frame_processor.paused = True
        audio_tensor = torch.randn(2, SAMPLES_PER_FRAME + 100)
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(audio_tensor, 48000)
        )

        frame = self._run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_recv_undersized_audio_then_silence(self):
        """If audio chunk is too small for a frame, recv() should return silence."""
        track = _make_track(channels=2, init_timestamp=False)
        small_audio = torch.randn(2, 100)
        track.frame_processor.get_audio = MagicMock(
            side_effect=_once(small_audio, 48000)
        )

        frame = self._run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_recv_accumulates_across_calls(self):
        """Multiple recv() calls with small chunks should accumulate until a frame is ready."""
        track = _make_track(channels=2, init_timestamp=False)
        chunk_size = 200  # Need 960 stereo samples = 1920 interleaved

        # Each recv() call gets one small chunk, then None
        call_idx = 0

        def get_audio_side_effect():
            nonlocal call_idx
            call_idx += 1
            # Alternate: odd calls return a chunk, even calls return None
            # This simulates one chunk available per recv() cycle
            if call_idx % 2 == 1:
                return (torch.randn(2, chunk_size), 48000)
            return (None, None)

        track.frame_processor.get_audio = MagicMock(side_effect=get_audio_side_effect)

        loop = asyncio.new_event_loop()
        try:
            got_real_frame = False
            for _ in range(10):
                frame = loop.run_until_complete(track.recv())
                raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
                if not np.all(raw == 0):
                    got_real_frame = True
                    break
        finally:
            loop.close()

        assert got_real_frame, "Should have accumulated enough for a real frame"

    def test_recv_drains_queue(self):
        """recv() should drain all available audio from the queue, not just one chunk."""
        track = _make_track(channels=2, init_timestamp=False)
        chunk_size = 200
        chunks_returned = 0

        def get_audio_side_effect():
            nonlocal chunks_returned
            if chunks_returned < 5:
                chunks_returned += 1
                return (torch.randn(2, chunk_size), 48000)
            return (None, None)

        track.frame_processor.get_audio = MagicMock(side_effect=get_audio_side_effect)

        # Single recv() call should drain all 5 chunks
        self._run(track.recv())
        assert chunks_returned == 5, "recv() should drain all queued audio chunks"
        # Buffer should have 5 * 200 * 2 = 2000 interleaved samples
        # minus 1920 consumed for the frame = 80 remaining
        assert len(track._audio_buffer) == 80

    def test_recv_caps_buffer_at_max(self):
        """Buffer should be capped at AUDIO_MAX_BUFFER_SAMPLES to prevent unbounded growth."""
        from scope.server.tracks import AUDIO_MAX_BUFFER_SAMPLES

        track = _make_track(channels=2, init_timestamp=False)
        # Stuff the buffer with more than the max
        oversized = np.ones(AUDIO_MAX_BUFFER_SAMPLES * 2 + 5000, dtype=np.float32)
        track._audio_buffer = oversized

        # recv() should trim the buffer
        track.frame_processor.get_audio = MagicMock(return_value=(None, None))
        self._run(track.recv())
        # After trimming and consuming one frame, buffer should be under max
        assert len(track._audio_buffer) <= AUDIO_MAX_BUFFER_SAMPLES * 2
