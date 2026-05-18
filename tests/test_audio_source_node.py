"""Tests for the built-in AudioSource node and its WAV decoder."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from scope.core.nodes.builtins.audio_io import (
    SAMPLE_RATE,
    AudioSourceNode,
    _read_wav_float32,
)


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples to a 16-bit PCM WAV via stdlib ``wave``."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    n_channels = pcm.shape[1] if pcm.ndim == 2 else 1
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _write_float32(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Hand-roll an IEEE-float32 WAV (stdlib ``wave`` rejects format 3)."""
    if samples.ndim == 1:
        samples = samples[:, None]
    n_channels = samples.shape[1]
    data = samples.astype("<f4").tobytes()
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        n_channels,
        sample_rate,
        sample_rate * n_channels * 4,  # byte rate
        n_channels * 4,  # block align
        32,  # bits per sample
    )
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data))
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def test_read_wav_pcm16_stereo(tmp_path: Path) -> None:
    sr = 16000
    samples = np.random.uniform(-0.5, 0.5, (sr, 2)).astype(np.float32)
    wav = tmp_path / "pcm.wav"
    _write_pcm16(wav, samples, sr)

    data, decoded_sr = _read_wav_float32(str(wav))
    assert decoded_sr == sr
    assert data.shape == (sr, 2)
    # 16-bit quantisation noise at most ~1/32767.
    assert np.max(np.abs(data - samples)) < 1.5 / 32767


def test_read_wav_float32_mono(tmp_path: Path) -> None:
    sr = 22050
    samples = np.random.uniform(-0.9, 0.9, sr).astype(np.float32)
    wav = tmp_path / "float.wav"
    _write_float32(wav, samples, sr)

    data, decoded_sr = _read_wav_float32(str(wav))
    assert decoded_sr == sr
    assert data.shape == (sr, 1)
    np.testing.assert_allclose(data[:, 0], samples, atol=1e-6)


def test_read_wav_rejects_non_wav(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"NOT A WAV FILE")
    with pytest.raises(ValueError):
        _read_wav_float32(str(bad))


def test_audio_source_emits_full_clip(tmp_path: Path) -> None:
    """``execute`` returns the entire decoded clip as a (channels, samples) tensor."""
    sr = SAMPLE_RATE
    duration_s = 1.0
    samples = np.random.uniform(-0.3, 0.3, (int(sr * duration_s), 2)).astype(np.float32)
    wav = tmp_path / "clip.wav"
    _write_pcm16(wav, samples, sr)

    node = AudioSourceNode(node_id="audio_src")
    out = node.execute({}, file_id=str(wav), duration=duration_s)

    assert "audio" in out
    tensor, out_sr = out["audio"]
    assert out_sr == SAMPLE_RATE
    assert tensor.shape == (2, int(sr * duration_s))


def test_audio_source_missing_file_is_silent(tmp_path: Path) -> None:
    """Missing ``file_id`` returns ``{}`` instead of raising."""
    node = AudioSourceNode(node_id="audio_src")
    assert node.execute({}) == {}
    assert node.execute({}, file_id=str(tmp_path / "does_not_exist.wav")) == {}


def test_audio_source_duration_change_retrims(tmp_path: Path) -> None:
    """Changing only the duration must re-trim — the cache key includes it."""
    sr = SAMPLE_RATE
    samples = np.random.uniform(-0.3, 0.3, (sr * 4, 2)).astype(np.float32)
    wav = tmp_path / "clip.wav"
    _write_pcm16(wav, samples, sr)

    node = AudioSourceNode(node_id="audio_src")
    out_long = node.execute({}, file_id=str(wav), duration=4.0)
    assert out_long["audio"][0].shape[1] == 4 * sr

    out_short = node.execute({}, file_id=str(wav), duration=1.0)
    assert out_short["audio"][0].shape[1] == 1 * sr
