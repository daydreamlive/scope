"""Built-in audio I/O nodes: AudioSource (WAV file → audio stream).

Terminal audio output is handled by the regular Sink node: audio edges
into a Sink are routed straight to the WebRTC audio track via the
session's audio_output_queue, with no intermediate node needed.
"""

from __future__ import annotations

import logging
import os
import time
import wave
from typing import Any, ClassVar

import numpy as np
import torch

from ..base import BaseNode, NodeDefinition, NodeParam, NodePort

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
CHUNK_DURATION = 0.1  # 100ms chunks for streaming
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


class AudioSourceNode(BaseNode):
    """Load audio from a WAV file and stream it in 100ms chunks, looping."""

    node_type_id: ClassVar[str] = "audio.AudioSource"

    def __init__(self, node_id: str, config: dict[str, Any] | None = None):
        super().__init__(node_id, config)
        self._audio_data: np.ndarray | None = None
        self._position = 0
        self._loaded_file: str = ""
        self._last_call_time: float | None = None
        self._output_mode: str = "stream"  # "stream" or "full"

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Audio Source",
            category="audio",
            description="Load audio from a WAV file at 48kHz stereo.",
            continuous=True,
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
                NodeParam(
                    name="mode",
                    param_type="select",
                    default="full",
                    description="Output mode",
                    ui={"options": ["full", "stream"]},
                ),
            ],
        )

    def _load_audio(self, file_path: str, duration: float) -> None:
        """Load, decode, resample to 48kHz stereo, and clip to duration."""
        with wave.open(file_path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        data = data.reshape(-1, n_channels)
        if n_channels == 1:
            data = np.stack([data[:, 0], data[:, 0]], axis=-1)
        elif n_channels > 2:
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
        self._position = 0
        self._loaded_file = file_path
        logger.info(
            "AudioSource loaded: %s (%.1fs)",
            file_path,
            data.shape[1] / SAMPLE_RATE,
        )

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        file_id = kwargs.get("file_id", self.config.get("file_id", ""))
        duration = float(kwargs.get("duration", self.config.get("duration", 60.0)))
        # "full" = emit entire clip once (for batch DAGs); "stream" = 100ms chunks
        mode = kwargs.get("mode", self.config.get("mode", "stream"))

        if not file_id:
            return {}
        file_id = self._resolve_path(file_id)
        if not file_id:
            return {}

        if file_id != self._loaded_file:
            try:
                self._load_audio(file_id, duration)
            except Exception as e:
                logger.error("AudioSourceNode failed to load %s: %s", file_id, e)
                return {}

        if self._audio_data is None or self._audio_data.shape[1] == 0:
            return {}

        if mode == "full":
            return self._emit_full()
        return self._emit_chunk()

    @staticmethod
    def _resolve_path(file_id: str) -> str | None:
        """Resolve a file path, checking common locations for relative names."""
        if os.path.isabs(file_id) and os.path.exists(file_id):
            return file_id
        # Check relative to cwd
        if os.path.exists(file_id):
            return os.path.abspath(file_id)
        # Check Scope assets directory
        from pathlib import Path

        assets_dir = Path.home() / ".daydream-scope" / "assets"
        candidate = assets_dir / file_id
        if candidate.exists():
            return str(candidate)
        # Check installed plugin directories (e.g. ACEStep's scope_plugin/)
        try:
            from importlib.metadata import distributions

            for dist in distributions():
                eps = dist.entry_points
                scope_eps = [ep for ep in eps if ep.group == "scope"]
                if scope_eps:
                    loc = dist._path  # type: ignore[attr-defined]
                    if loc:
                        for parent in [loc.parent, loc.parent.parent]:
                            candidate = parent / file_id
                            if candidate.exists():
                                return str(candidate)
        except Exception:
            pass
        logger.warning("AudioSource: file not found: %s", file_id)
        return None

    def _emit_full(self) -> dict[str, Any]:
        audio_tensor = torch.from_numpy(self._audio_data.copy())
        return self._wrap(audio_tensor)

    def _emit_chunk(self) -> dict[str, Any]:
        # Pace to real-time
        now = time.monotonic()
        if self._last_call_time is not None:
            elapsed = now - self._last_call_time
            if elapsed < CHUNK_DURATION * 0.8:
                time.sleep(CHUNK_DURATION - elapsed)
        self._last_call_time = time.monotonic()

        total = self._audio_data.shape[1]
        chunk = np.zeros((self._audio_data.shape[0], CHUNK_SAMPLES), dtype=np.float32)
        remaining = CHUNK_SAMPLES
        offset = 0
        while remaining > 0:
            avail = min(remaining, total - self._position)
            chunk[:, offset : offset + avail] = self._audio_data[
                :, self._position : self._position + avail
            ]
            self._position = (self._position + avail) % total
            offset += avail
            remaining -= avail
        return self._wrap(torch.from_numpy(chunk))

    @staticmethod
    def _wrap(audio_tensor: torch.Tensor) -> dict[str, Any]:
        """Emit audio as a (tensor, sample_rate) tuple."""
        return {"audio": (audio_tensor, SAMPLE_RATE)}
