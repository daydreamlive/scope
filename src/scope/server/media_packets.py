"""Timestamp-aware transport wrappers for media flowing through the server."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import torch


@dataclass(frozen=True)
class MediaTimestamp:
    """Presentation timestamp metadata attached to media payloads."""

    pts: int | None = None
    time_base: Fraction | None = None

    @property
    def is_valid(self) -> bool:
        return self.pts is not None and self.time_base is not None


@dataclass(frozen=True)
class VideoPacket:
    """Video tensor plus optional timestamp metadata."""

    tensor: torch.Tensor
    timestamp: MediaTimestamp = MediaTimestamp()


@dataclass(frozen=True)
class AudioPacket:
    """Audio tensor chunk plus sample rate and optional timestamp metadata."""

    audio: torch.Tensor | None
    sample_rate: int
    timestamp: MediaTimestamp = MediaTimestamp()

    @property
    def is_flush(self) -> bool:
        return self.audio is None and self.sample_rate == -1


def ensure_video_packet(item: Any) -> VideoPacket:
    """Upgrade legacy queue payloads (tensor) to VideoPacket."""
    # Some sink/record paths still enqueue plain tensors instead of VideoPacket.
    # Once queue get/put paths consistently traffic in VideoPacket/AudioPacket,
    # this compatibility helper can be removed entirely.
    if isinstance(item, VideoPacket):
        return item
    return VideoPacket(tensor=item)


def ensure_audio_packet(
    item: tuple[torch.Tensor | None, int] | AudioPacket,
) -> AudioPacket:
    """Upgrade legacy audio payload tuples to AudioPacket."""
    if isinstance(item, AudioPacket):
        return item
    audio, sample_rate = item
    return AudioPacket(audio=audio, sample_rate=sample_rate)


def audio_packet_from_node_output(value: Any) -> AudioPacket | None:
    """Build an ``AudioPacket`` from a graph node's audio output port value.

    Accepts the two shapes that built-in and plugin nodes emit:
      * ``(tensor, sample_rate)`` tuples (e.g. ``AudioSourceNode``).
      * Objects with ``.waveform`` / ``.sample_rate`` / optional
        ``.start_sample`` attributes (e.g. ACEStep ``StreamVAEDecode``,
        whose ``start_sample`` is forwarded as PTS so ``AudioProcessingTrack``
        can trim overlapping windows downstream).

    Normalizes the tensor for ``AudioProcessingTrack`` consumption: moves
    to CPU, drops a leading ``(1, C, T)`` batch dim, and upcasts
    bfloat16/float16 to float32 so the subsequent ``.numpy()`` call works.
    Returns ``None`` when no audio tensor can be extracted.
    """
    start_sample: int | None = None
    if isinstance(value, tuple) and len(value) == 2:
        audio_tensor, audio_sr = value
    else:
        audio_tensor = getattr(value, "waveform", None)
        audio_sr = getattr(value, "sample_rate", 48000)
        start_sample = getattr(value, "start_sample", None)

    if audio_tensor is None:
        return None

    if isinstance(audio_tensor, torch.Tensor):
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.detach().cpu()
        if audio_tensor.dim() == 3 and audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.dtype in (torch.bfloat16, torch.float16):
            audio_tensor = audio_tensor.float()

    timestamp = (
        MediaTimestamp(pts=int(start_sample), time_base=Fraction(1, int(audio_sr)))
        if start_sample is not None and audio_sr
        else MediaTimestamp()
    )
    return AudioPacket(
        audio=audio_tensor, sample_rate=int(audio_sr), timestamp=timestamp
    )
