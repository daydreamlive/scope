import queue

import pytest

from scope.server import webrtc
from scope.server.recording_coordinator import RecordingCoordinator


@pytest.mark.anyio
async def test_start_recording_drops_stale_record_queue_frames(monkeypatch):
    coordinator = RecordingCoordinator()
    rec_q: queue.Queue = queue.Queue(maxsize=10)
    rec_q.put_nowait("stale-1")
    rec_q.put_nowait("stale-2")
    rec_q.put_nowait("stale-3")
    coordinator.setup_queues({"record": rec_q})

    captured = {}

    class FakeQueueVideoTrack:
        def __init__(self, frame_queue, fps: float = 30.0):
            captured["queue_size_on_init"] = frame_queue.qsize()
            captured["fps"] = fps

        def stop(self):
            return

    class FakeRecordingManager:
        def __init__(self, video_track=None):
            self.is_recording_started = False
            captured["track_instance"] = video_track

        async def start_recording(self):
            captured["manager_started"] = True

    monkeypatch.setattr("scope.server.tracks.QueueVideoTrack", FakeQueueVideoTrack)
    monkeypatch.setattr("scope.server.recording.RecordingManager", FakeRecordingManager)

    ok = await coordinator.start_recording("record", fps=29.97)

    assert ok is True
    assert rec_q.qsize() == 0
    assert captured["queue_size_on_init"] == 0
    assert captured["fps"] == 29.97
    assert captured["manager_started"] is True


def test_record_nodes_are_not_attached_as_live_webrtc_outputs(monkeypatch):
    constructed_sinks = []

    def fake_sink_output_track(*, frame_processor, sink_node_id):
        constructed_sinks.append((frame_processor, sink_node_id))
        return {"kind": "sink", "id": sink_node_id}

    monkeypatch.setattr(webrtc, "SinkOutputTrack", fake_sink_output_track)

    frame_processor = object()
    tracks = webrtc._build_extra_output_tracks(
        frame_processor=frame_processor,
        sink_node_ids=["preview", "secondary"],
        record_node_ids=["record"],
    )

    assert tracks == [{"kind": "sink", "id": "secondary"}]
    assert constructed_sinks == [(frame_processor, "secondary")]
