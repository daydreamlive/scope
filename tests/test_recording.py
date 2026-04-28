import asyncio
import fractions
import os
import queue

import av
import numpy as np
import pytest
import torch
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame

from scope.server.media_packets import VideoPacket
from scope.server.recording import (
    RECORDING_MAX_FPS,
    RecordingManager,
    ensure_even_video_frame,
)
from scope.server.tracks import QueueVideoTrack


class _SyntheticTimestampTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        *,
        frame_count: int,
        pts_step: int,
        time_base: fractions.Fraction,
        width: int = 160,
        height: int = 120,
    ):
        super().__init__()
        self._frame_count = frame_count
        self._pts_step = pts_step
        self._time_base = time_base
        self._width = width
        self._height = height
        self._index = 0

    @property
    def delivered(self) -> int:
        return self._index

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError
        if self._index >= self._frame_count:
            raise MediaStreamError
        frame = VideoFrame.from_ndarray(
            np.zeros((self._height, self._width, 3), dtype=np.uint8),
            format="rgb24",
        )
        frame.pts = self._index * self._pts_step
        frame.time_base = self._time_base
        self._index += 1
        await asyncio.sleep(0)
        return frame


def _read_video_dts(path: str) -> list[int]:
    with av.open(path) as container:
        stream = container.streams.video[0]
        dts = []
        for packet in container.demux(stream):
            if packet.dts is None or packet.size == 0:
                continue
            dts.append(packet.dts)
        return dts


@pytest.mark.anyio
async def test_recording_preserves_distinct_granular_timestamps() -> None:
    # 90kHz timeline at ~31fps used to collapse to duplicate mux timestamps
    # when coerced onto a 30fps stream time base.
    track = _SyntheticTimestampTrack(
        frame_count=120,
        pts_step=int(90_000 / 31),
        time_base=fractions.Fraction(1, 90_000),
    )
    manager = RecordingManager(video_track=track)
    recording_file = None
    try:
        await manager.start_recording()
        for _ in range(200):
            if track.delivered >= 120:
                break
            await asyncio.sleep(0.01)
        recording_file = manager.recording_file
        await manager.stop_recording()

        assert recording_file is not None
        assert os.path.exists(recording_file)
        assert os.path.getsize(recording_file) > 48
        dts = _read_video_dts(recording_file)
        assert len(dts) > 8
        assert len(set(dts)) == len(dts)
        assert all(b > a for a, b in zip(dts, dts[1:], strict=False))
    finally:
        if recording_file and os.path.exists(recording_file):
            os.remove(recording_file)


@pytest.mark.anyio
async def test_gray_like_synthesized_fallback_timestamps_remain_valid() -> None:
    # Simulate gray-like behavior: no output timestamps provided by pipeline,
    # so QueueVideoTrack synthesizes them using fps. This previously failed
    # when fps > RECORDING_MAX_FPS due to mux timestamp collapse.
    rec_q: queue.Queue = queue.Queue()
    frame = torch.zeros((1, 120, 160, 3), dtype=torch.uint8)
    for _ in range(240):
        rec_q.put_nowait(VideoPacket(tensor=frame))

    track = QueueVideoTrack(rec_q, fps=RECORDING_MAX_FPS * 2)
    manager = RecordingManager(video_track=track)
    recording_file = None
    try:
        await manager.start_recording()
        await asyncio.sleep(1.0)
        recording_file = manager.recording_file
        await manager.stop_recording()

        assert recording_file is not None
        assert os.path.exists(recording_file)
        assert os.path.getsize(recording_file) > 48
        dts = _read_video_dts(recording_file)
        assert len(dts) > 8
        assert len(set(dts)) == len(dts)
        assert all(b > a for a, b in zip(dts, dts[1:], strict=False))
    finally:
        if recording_file and os.path.exists(recording_file):
            os.remove(recording_file)


def test_ensure_even_video_frame_preserves_pts_and_time_base() -> None:
    frame = VideoFrame.from_ndarray(
        np.zeros((3, 5, 3), dtype=np.uint8),
        format="rgb24",
    )
    frame.pts = 321
    frame.time_base = fractions.Fraction(1, 90_000)

    padded = ensure_even_video_frame(frame)

    assert padded.width == 6
    assert padded.height == 4
    assert padded.pts == 321
    assert padded.time_base == fractions.Fraction(1, 90_000)
