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
