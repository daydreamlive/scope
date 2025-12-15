import logging
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..interface import Pipeline, Requirements
from ..schema import VibeVoiceConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig


logger = logging.getLogger(__name__)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Naive linear resampling to match WebRTC's expected sample rate."""
    if source_rate == target_rate or audio.size == 0:
        return audio

    # Generate target indices and interpolate
    duration = audio.shape[0] / source_rate
    target_length = int(duration * target_rate)
    target_positions = np.linspace(0, audio.shape[0] - 1, num=target_length)
    resampled = np.interp(target_positions, np.arange(audio.shape[0]), audio)
    return resampled.astype(audio.dtype)


def _generate_fallback_tone(duration_seconds: float, sample_rate: int) -> np.ndarray:
    """Generate a short fallback tone so the player has audible output."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440Hz sine wave scaled to int16
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    return (tone * 32767).astype(np.int16)


class VibeVoicePipeline(Pipeline):
    """Lightweight text-to-speech stub that streams a wav file in chunks."""

    target_sample_rate = 48_000
    default_chunk_size = 960  # 20ms @ 48k, aligns with common WebRTC audio pacing

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VibeVoiceConfig

    def __init__(self, audio_path: str | None = None, chunk_size: int | None = None):
        # Allow overriding the audio file; otherwise use well-known locations
        self._audio_path = Path(audio_path) if audio_path else None
        self.chunk_size = chunk_size or self.default_chunk_size

        self._audio_buffer: np.ndarray | None = None
        self.sample_rate: int = self.target_sample_rate
        self._position = 0
        self._lock = threading.Lock()

        # Track the latest text for logging/debugging (not used in this stub)
        self._last_text: str | None = None

    # Interface compatibility -------------------------------------------------
    def prepare(self, **kwargs) -> Requirements:
        """Load/refresh audio buffer and reset streaming state."""
        text = kwargs.get("text")
        prompts = kwargs.get("prompts") or []
        if text:
            self._last_text = text
        elif prompts:
            # Use the first prompt's text as the input text
            first = prompts[0]
            if isinstance(first, dict):
                self._last_text = first.get("text")
            else:
                self._last_text = getattr(first, "text", None)

        chunk_size = kwargs.get("chunk_size") or self.chunk_size
        with self._lock:
            self.chunk_size = chunk_size
            self._load_audio()
            self._position = 0

        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> torch.Tensor | None:
        """Return the next chunk of audio as a tensor in [-1, 1] range."""
        reset_requested = kwargs.get("init_cache") or kwargs.get("reset_cache")
        if reset_requested:
            self.prepare(**kwargs)

        with self._lock:
            if self._audio_buffer is None or self._audio_buffer.size == 0:
                return None

            if self._position >= self._audio_buffer.shape[0]:
                return None

            end = min(self._position + self.chunk_size, self._audio_buffer.shape[0])
            chunk = self._audio_buffer[self._position : end]
            self._position = end

        # Normalize to [-1, 1] float32 for downstream processing
        chunk_float = np.clip(chunk.astype(np.float32) / 32767.0, -1.0, 1.0)
        return torch.from_numpy(chunk_float)

    # Internal helpers --------------------------------------------------------
    def _load_audio(self):
        """Load the audio from disk (or generate a fallback)."""
        candidate_paths = []
        if self._audio_path:
            candidate_paths.append(self._audio_path)
        # Prefer the external VibeVoice checkout if it exists
        candidate_paths.append(Path("/home/user/VibeVoice/output.wav"))
        # Fallback: look next to this file
        candidate_paths.append(Path(__file__).parent / "output.wav")

        path_to_use = next((p for p in candidate_paths if p.is_file()), None)

        if path_to_use is None:
            logger.warning(
                "VibeVoicePipeline: output.wav not found, generating fallback tone"
            )
            self.sample_rate = self.target_sample_rate
            self._audio_buffer = _generate_fallback_tone(3.0, self.sample_rate)
            return

        try:
            with wave.open(str(path_to_use), "rb") as wf:
                source_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                num_frames = wf.getnframes()
                audio_bytes = wf.readframes(num_frames)

            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            if num_channels > 1:
                audio = audio.reshape(-1, num_channels).mean(axis=1).astype(np.int16)

            if source_rate != self.target_sample_rate:
                audio = _resample_audio(audio, source_rate, self.target_sample_rate)

            self._audio_buffer = audio
            self.sample_rate = self.target_sample_rate
            logger.info(
                "VibeVoicePipeline loaded audio from %s (src_rate=%s, samples=%s)",
                path_to_use,
                source_rate,
                audio.shape[0],
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load audio for VibeVoice: %s", exc)
            self.sample_rate = self.target_sample_rate
            self._audio_buffer = _generate_fallback_tone(3.0, self.sample_rate)
