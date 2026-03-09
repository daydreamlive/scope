"""Adversarial tests for AudioProcessingTrack's numpy-based audio buffer.

Exercises edge cases in interleaving, buffering, resampling, and frame
construction to ensure the vectorized numpy path handles degenerate inputs
without crashing or producing corrupt audio frames.
"""

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import torch
from av import AudioFrame

from scope.server.tracks import AUDIO_CLOCK_RATE, AUDIO_PTIME, AudioProcessingTrack

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


# ---------------------------------------------------------------------------
# Interleaving correctness
# ---------------------------------------------------------------------------


class TestInterleaving:
    """Verify that channel interleaving produces the correct sample order."""

    def test_stereo_interleave_order(self):
        """L/R samples must alternate: [L0, R0, L1, R1, ...]."""
        track = _make_track(channels=2)

        left = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        right = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        audio = np.stack([left, right])  # (2, 3)

        # Simulate the interleave path from recv()
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        expected = np.array([1, 10, 2, 20, 3, 30], dtype=np.float32)
        np.testing.assert_array_equal(interleaved, expected)

    def test_mono_passthrough(self):
        """Single channel should flatten without interleaving artifacts."""
        track = _make_track(channels=1)
        audio = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # (1, 3)
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(interleaved, expected)

    def test_many_channels_interleave(self):
        """Verify interleaving with >2 channels (future-proofing)."""
        audio = np.arange(12, dtype=np.float32).reshape(3, 4)
        # Fortran order: column-major, so [col0_row0, col0_row1, col0_row2, col1_row0, ...]
        interleaved = np.ravel(audio, order="F")
        # First 3 samples should be channel 0/1/2 of sample 0
        assert interleaved[0] == audio[0, 0]
        assert interleaved[1] == audio[1, 0]
        assert interleaved[2] == audio[2, 0]


# ---------------------------------------------------------------------------
# Buffer accumulation and frame extraction
# ---------------------------------------------------------------------------


class TestBufferAccumulation:
    """Test that the numpy buffer correctly accumulates and drains."""

    def test_exact_frame_size(self):
        """Buffer with exactly one frame's worth of samples should produce a frame."""
        track = _make_track(channels=2)
        samples_needed = SAMPLES_PER_FRAME * 2
        track._audio_buffer = np.zeros(samples_needed, dtype=np.float32)

        assert len(track._audio_buffer) >= samples_needed
        frame_samples = track._audio_buffer[:samples_needed]
        track._audio_buffer = track._audio_buffer[samples_needed:]

        assert len(frame_samples) == samples_needed
        assert len(track._audio_buffer) == 0

    def test_undersized_buffer_returns_nothing(self):
        """Buffer smaller than one frame should not yield a frame."""
        track = _make_track(channels=2)
        samples_needed = SAMPLES_PER_FRAME * 2
        track._audio_buffer = np.zeros(samples_needed - 1, dtype=np.float32)

        assert len(track._audio_buffer) < samples_needed

    def test_multiple_frames_from_large_chunk(self):
        """A large audio chunk should allow draining multiple frames."""
        track = _make_track(channels=2)
        samples_needed = SAMPLES_PER_FRAME * 2
        # 3.5 frames worth
        total = int(samples_needed * 3.5)
        track._audio_buffer = np.random.randn(total).astype(np.float32)

        frames_extracted = 0
        while len(track._audio_buffer) >= samples_needed:
            track._audio_buffer = track._audio_buffer[samples_needed:]
            frames_extracted += 1

        assert frames_extracted == 3
        assert len(track._audio_buffer) == total - 3 * samples_needed

    def test_successive_small_chunks_fill_frame(self):
        """Many tiny chunks should accumulate until a full frame is available."""
        track = _make_track(channels=2)
        samples_needed = SAMPLES_PER_FRAME * 2
        chunk_size = 64  # much smaller than 1920
        chunks_needed = (samples_needed // chunk_size) + 1

        for _ in range(chunks_needed):
            chunk = np.zeros(chunk_size, dtype=np.float32)
            track._audio_buffer = np.concatenate([track._audio_buffer, chunk])

        assert len(track._audio_buffer) >= samples_needed

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

        # Extract int16 data from the frame
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
# Degenerate / adversarial audio inputs
# ---------------------------------------------------------------------------


class TestAdversarialInputs:
    """Inputs that could break naive implementations."""

    def test_zero_length_audio(self):
        """Empty audio tensor should not crash or corrupt the buffer."""
        track = _make_track(channels=2)
        audio = np.zeros((2, 0), dtype=np.float32)
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        track._audio_buffer = np.concatenate([track._audio_buffer, interleaved])

        assert len(track._audio_buffer) == 0

    def test_single_sample_stereo(self):
        """A single stereo sample (2, 1) should produce 2 interleaved values."""
        track = _make_track(channels=2)
        audio = np.array([[0.5], [-0.5]], dtype=np.float32)
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        track._audio_buffer = np.concatenate([track._audio_buffer, interleaved])

        assert len(track._audio_buffer) == 2
        np.testing.assert_array_almost_equal(track._audio_buffer, [0.5, -0.5])

    def test_nan_values_dont_crash(self):
        """NaN in audio should not crash frame creation (will produce garbage, but no exception)."""
        track = _make_track(channels=2)
        n = SAMPLES_PER_FRAME * 2
        samples = np.full(n, np.nan, dtype=np.float32)
        # Should not raise
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

    def test_very_large_chunk_doesnt_oom(self):
        """A large audio chunk (10 seconds stereo @ 48kHz) should not cause issues."""
        track = _make_track(channels=2)
        ten_seconds = 48000 * 10
        audio = np.random.randn(2, ten_seconds).astype(np.float32)
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        track._audio_buffer = np.concatenate([track._audio_buffer, interleaved])

        assert len(track._audio_buffer) == ten_seconds * 2

    def test_float64_input_cast(self):
        """float64 audio should be handled (astype in interleave path)."""
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        interleaved = np.ravel(audio, order="F").astype(np.float32)
        assert interleaved.dtype == np.float32
        np.testing.assert_array_almost_equal(interleaved, [0.1, 0.3, 0.2, 0.4])

    def test_1d_audio_reshape(self):
        """1D audio tensor (mono without channel dim) should be reshaped to (1, N)."""
        audio_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        assert audio_np.shape == (1, 3)

    def test_dc_offset_preserved(self):
        """A constant DC offset should survive interleave + frame creation intact."""
        track = _make_track(channels=1)
        dc = 0.5
        n = SAMPLES_PER_FRAME
        samples = np.full(n, dc, dtype=np.float32)
        frame = track._create_audio_frame(samples)

        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        expected = int(dc * 32767)
        assert np.all(raw == expected)


# ---------------------------------------------------------------------------
# Channel conversion
# ---------------------------------------------------------------------------


class TestChannelConversion:
    """Test mono <-> stereo conversion paths."""

    def test_mono_to_stereo_duplication(self):
        """Mono (1, N) expanded to stereo (2, N) should duplicate the channel."""
        audio = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        channels = 2
        if audio.shape[0] == 1 and channels == 2:
            audio = np.vstack([audio, audio])

        assert audio.shape == (2, 3)
        np.testing.assert_array_equal(audio[0], audio[1])

    def test_stereo_to_mono_averaging(self):
        """Stereo (2, N) collapsed to mono should average channels."""
        audio = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        channels = 1
        if audio.shape[0] == 2 and channels == 1:
            audio = audio.mean(axis=0, keepdims=True)

        assert audio.shape == (1, 2)
        np.testing.assert_array_almost_equal(audio[0], [0.5, 0.5])

    def test_stereo_to_mono_phase_cancellation(self):
        """Opposite-phase stereo should cancel to silence when averaged."""
        audio = np.array([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
        audio = audio.mean(axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(audio[0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:
    """Test the FFT-based resampler with adversarial inputs."""

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
        # Allow some edge effects but the bulk should be close to dc
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
        """recv() should process a torch tensor from the queue and return an AudioFrame."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(2, n_samples)
        track.frame_processor.get_audio = MagicMock(return_value=(audio_tensor, 48000))

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.samples == SAMPLES_PER_FRAME

    def test_recv_with_resampling(self):
        """recv() should resample 24kHz audio to 48kHz and still produce a valid frame."""
        track = _make_track(channels=2, init_timestamp=False)
        n_input = SAMPLES_PER_FRAME  # more than enough after upsampling
        audio_tensor = torch.randn(2, n_input)
        track.frame_processor.get_audio = MagicMock(return_value=(audio_tensor, 24000))

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == AUDIO_CLOCK_RATE

    def test_recv_mono_input_stereo_output(self):
        """Mono audio should be upmixed to stereo in recv()."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(1, n_samples)  # mono
        track.frame_processor.get_audio = MagicMock(return_value=(audio_tensor, 48000))

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)
        assert frame.layout.name == "stereo"

    def test_recv_1d_tensor(self):
        """A 1D tensor (no channel dim) should be handled as mono."""
        track = _make_track(channels=2, init_timestamp=False)
        n_samples = SAMPLES_PER_FRAME + 100
        audio_tensor = torch.randn(n_samples)  # 1D
        track.frame_processor.get_audio = MagicMock(return_value=(audio_tensor, 48000))

        frame = self._run(track.recv())
        assert isinstance(frame, AudioFrame)

    def test_recv_paused_returns_silence(self):
        """When paused, recv() should return silence regardless of queued audio."""
        track = _make_track(channels=2, init_timestamp=False)
        track.frame_processor.paused = True
        audio_tensor = torch.randn(2, SAMPLES_PER_FRAME + 100)
        track.frame_processor.get_audio = MagicMock(return_value=(audio_tensor, 48000))

        frame = self._run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        assert np.all(raw == 0)

    def test_recv_undersized_audio_then_silence(self):
        """If audio chunk is too small for a frame, recv() should return silence
        until enough accumulates."""
        track = _make_track(channels=2, init_timestamp=False)
        small_audio = torch.randn(2, 100)
        track.frame_processor.get_audio = MagicMock(return_value=(small_audio, 48000))

        frame = self._run(track.recv())
        raw = np.frombuffer(bytes(frame.planes[0]), dtype=np.int16)
        # Should be silence since buffer is undersized
        assert np.all(raw == 0)

    def test_recv_accumulates_across_calls(self):
        """Multiple undersized recv() calls should accumulate until a frame is ready."""
        track = _make_track(channels=2, init_timestamp=False)
        chunk_size = 200  # Need 960 stereo samples = 1920 interleaved

        call_count = 0

        def get_audio_side_effect():
            nonlocal call_count
            call_count += 1
            return (torch.randn(2, chunk_size), 48000)

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
