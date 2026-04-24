import queue
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch
from av import VideoFrame

from scope.server.media_packets import MediaTimestamp, VideoPacket
from scope.server.recording import RecordingManager, ensure_even_video_frame
from scope.server.recording_coordinator import RecordingCoordinator
from scope.server.sink_manager import SinkManager
from scope.server.tracks import QueueVideoTrack


class _DummyTrack:
    kind = "video"

    def stop(self):
        return None


class _DummyRelay:
    def __init__(self, relayed_track):
        self._relayed_track = relayed_track

    def subscribe(self, track):
        assert track is not None
        return self._relayed_track


class _FakeVideoFrame:
    pts = 33
    time_base = Fraction(1, 24)

    def to_ndarray(self, format: str):
        assert format == "rgb24"
        return torch.zeros((2, 2, 3), dtype=torch.uint8).numpy()


def test_recording_manager_uses_relay_track_without_env_toggle():
    source_track = _DummyTrack()
    relayed_track = _DummyTrack()
    manager = RecordingManager(video_track=source_track)
    manager.set_relay(_DummyRelay(relayed_track))

    track = manager._create_recording_track()

    assert track is relayed_track


def test_recording_coordinator_get_packet_preserves_timestamp():
    coordinator = RecordingCoordinator()
    coordinator.setup_queues(["record"])
    frame = torch.ones((1, 2, 2, 3), dtype=torch.uint8)
    packet = VideoPacket(
        tensor=frame,
        timestamp=MediaTimestamp(pts=9, time_base=Fraction(1, 30)),
    )
    coordinator.put("record", packet)

    result = coordinator.get_packet("record")

    assert result is not None
    assert torch.equal(result.tensor, frame.squeeze(0))
    assert result.timestamp == packet.timestamp


def test_sink_manager_put_to_record_enqueues_video_packet_with_timestamp():
    manager = object.__new__(SinkManager)
    record_queue = queue.Queue()
    manager._recording = SimpleNamespace(_record_queues={"record": record_queue})

    manager.put_to_record("record", _FakeVideoFrame())

    packet = record_queue.get_nowait()
    assert isinstance(packet, VideoPacket)
    assert packet.timestamp == MediaTimestamp(pts=33, time_base=Fraction(1, 24))


def test_ensure_even_video_frame_pads_odd_dimensions():
    frame = VideoFrame.from_ndarray(
        torch.zeros((3, 5, 3), dtype=torch.uint8).numpy(),
        format="rgb24",
    )
    frame.pts = 12
    frame.time_base = Fraction(1, 30)

    even_frame = ensure_even_video_frame(frame)

    assert even_frame.width == 6
    assert even_frame.height == 4
    assert even_frame.pts == 12
    assert even_frame.time_base == Fraction(1, 30)


@pytest.mark.anyio
async def test_queue_video_track_preserves_timestamp_and_pads_dimensions():
    packet = VideoPacket(
        tensor=torch.zeros((1, 3, 5, 3), dtype=torch.uint8),
        timestamp=MediaTimestamp(pts=120, time_base=Fraction(1, 60)),
    )
    frame_queue: queue.Queue = queue.Queue()
    frame_queue.put_nowait(packet)
    track = QueueVideoTrack(frame_queue, fps=30.0)

    frame = await track.recv()

    assert frame.width == 6
    assert frame.height == 4
    assert frame.pts == 120
    assert frame.time_base == Fraction(1, 60)
