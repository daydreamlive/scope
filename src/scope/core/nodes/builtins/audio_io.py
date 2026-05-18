"""Built-in audio I/O nodes: AudioSource (load a WAV file once).

Terminal audio output is handled by the regular Sink node: audio edges
into a Sink are routed straight to the WebRTC audio track via the
session's audio_output_queue, with no intermediate node needed.
"""

from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from ..base import BaseNode, NodeDefinition, NodeParam, NodePort

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000


def _read_wav_float32(path: str) -> tuple[np.ndarray, int]:
    """Parse a WAV file into float32 samples without the stdlib ``wave``
    module, which rejects IEEE-float (format 3) files.

    Returns (data, sample_rate) where ``data`` has shape (samples, channels).
    Supports formats 1 (PCM int) and 3 (IEEE float) — the two common cases.
    WAVE_FORMAT_EXTENSIBLE (0xFFFE) is unwrapped to its underlying format.
    """
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Not a WAV file: {path}")

        fmt_code: int | None = None
        n_channels = 1
        sample_rate = 0
        bits_per_sample = 0
        pcm_bytes = b""

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            chunk_data = f.read(chunk_size)
            if chunk_size % 2 == 1:
                f.read(1)  # RIFF pads odd-sized chunks

            if chunk_id == b"fmt " and len(chunk_data) >= 16:
                (
                    fmt_code,
                    n_channels,
                    sample_rate,
                    _byte_rate,
                    _block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", chunk_data[:16])
                # Unwrap WAVE_FORMAT_EXTENSIBLE: real format is first 2 bytes of the GUID.
                if fmt_code == 0xFFFE and len(chunk_data) >= 26:
                    fmt_code = struct.unpack("<H", chunk_data[24:26])[0]
            elif chunk_id == b"data":
                pcm_bytes = chunk_data

        if fmt_code is None or not pcm_bytes or sample_rate <= 0:
            raise ValueError(f"WAV missing fmt/data chunk: {path}")

        if fmt_code == 1:  # PCM integer
            if bits_per_sample == 16:
                samples = (
                    np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
                )
            elif bits_per_sample == 32:
                samples = (
                    np.frombuffer(pcm_bytes, dtype="<i4").astype(np.float32)
                    / 2147483648.0
                )
            elif bits_per_sample == 8:
                samples = (
                    np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32) / 128.0
                    - 1.0
                )
            elif bits_per_sample == 24:
                raw = np.frombuffer(pcm_bytes, dtype=np.uint8).reshape(-1, 3)
                as32 = (
                    raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16)
                )
                # Sign-extend the 24-bit value.
                as32 = np.where(as32 & 0x800000, as32 | ~0xFFFFFF, as32)
                samples = as32.astype(np.float32) / 8388608.0
            else:
                raise ValueError(f"Unsupported PCM bit depth: {bits_per_sample}")
        elif fmt_code == 3:  # IEEE float
            if bits_per_sample == 32:
                samples = np.frombuffer(pcm_bytes, dtype="<f4").astype(
                    np.float32, copy=True
                )
            elif bits_per_sample == 64:
                samples = np.frombuffer(pcm_bytes, dtype="<f8").astype(np.float32)
            else:
                raise ValueError(f"Unsupported float bit depth: {bits_per_sample}")
        else:
            raise ValueError(f"Unsupported WAV format code: {fmt_code}")

        samples = samples.reshape(-1, n_channels)
        return samples, sample_rate


class AudioSourceNode(BaseNode):
    """Load an audio file and emit the entire clip once per (file_id, duration)."""

    node_type_id: ClassVar[str] = "audio.AudioSource"

    def __init__(self, node_id: str, config: dict[str, Any] | None = None):
        super().__init__(node_id, config)
        self._audio_data: np.ndarray | None = None
        self._loaded_file: str = ""
        self._loaded_duration: float = 0.0

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Audio Source",
            category="audio",
            description="Load audio from a WAV file at 48kHz stereo.",
            # continuous=False so NodeProcessor only re-runs us when a
            # parameter actually changes; otherwise the worker would call
            # execute() every tick and either flood the graph (on success)
            # or flood the log (on missing-file).
            continuous=False,
            inputs=[],
            outputs=[
                NodePort(name="audio", port_type="audio", description="Audio waveform"),
            ],
            params=[
                NodeParam(
                    name="file_id",
                    param_type="string",
                    default="",
                    description="Audio file path",
                ),
                NodeParam(
                    name="duration",
                    param_type="number",
                    default=15.0,
                    description="Duration (s)",
                    ui={"min": 1, "max": 600, "step": 1},
                ),
            ],
        )

    def _load_audio(self, file_path: str, duration: float) -> None:
        """Load, decode, resample to 48kHz stereo, and clip to duration."""
        data, sr = _read_wav_float32(file_path)  # (samples, channels)

        if data.shape[1] == 1:
            data = np.concatenate([data, data], axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
        data = data.T  # (channels, samples)

        if sr != SAMPLE_RATE and sr > 0:
            num_samples = data.shape[1]
            new_len = int(num_samples * SAMPLE_RATE / sr)
            old_indices = np.linspace(0, num_samples - 1, new_len)
            resampled = np.zeros((data.shape[0], new_len), dtype=np.float32)
            for ch in range(data.shape[0]):
                resampled[ch] = np.interp(old_indices, np.arange(num_samples), data[ch])
            data = resampled

        max_samples = int(duration * SAMPLE_RATE)
        if data.shape[1] > max_samples:
            data = data[:, :max_samples]

        self._audio_data = data
        self._loaded_file = file_path
        self._loaded_duration = duration
        logger.info(
            "AudioSource loaded: %s (%.1fs)",
            file_path,
            data.shape[1] / SAMPLE_RATE,
        )

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        file_id = kwargs.get("file_id", "")
        if not file_id:
            return {}
        resolved = self._resolve_path(file_id)
        if not resolved:
            return {}

        duration = float(kwargs.get("duration", 15.0))
        # Cache key includes duration: a duration change must re-trim
        # (or re-decode if duration grows past the current clip).
        if resolved != self._loaded_file or duration != self._loaded_duration:
            try:
                self._load_audio(resolved, duration)
            except Exception as e:
                logger.error("AudioSourceNode failed to load %s: %s", resolved, e)
                return {}

        if self._audio_data is None or self._audio_data.shape[1] == 0:
            return {}
        return {"audio": (torch.from_numpy(self._audio_data), SAMPLE_RATE)}

    @staticmethod
    def _resolve_path(file_id: str) -> str | None:
        """Resolve a file path; falls back to ``~/.daydream-scope/assets``."""
        if os.path.exists(file_id):
            return os.path.abspath(file_id)
        candidate = Path.home() / ".daydream-scope" / "assets" / file_id
        if candidate.exists():
            return str(candidate)
        logger.warning("AudioSource: file not found: %s", file_id)
        return None
