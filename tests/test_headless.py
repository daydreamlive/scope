import pytest
import torch

from scope.server.headless import HeadlessMediaSink, HeadlessSession


class _SingleFrameProcessor:
    def __init__(self, frame: torch.Tensor):
        self.running = True
        self._frame = frame
        self._served = False

    def get_sink_node_ids(self):
        return []

    def get_from_sink(self, sink_node_id):
        return None

    def get(self):
        if self._served:
            return None
        self._served = True
        self.running = False
        return self._frame

    def get_audio(self):
        return None, None

    def stop(self):
        self.running = False


class _SinkOnlyFrameProcessor:
    def __init__(self, frames_by_sink: dict[str, torch.Tensor]):
        self.running = True
        self._sink_ids = list(frames_by_sink.keys())
        self._frames_by_sink = dict(frames_by_sink)
        self.get_calls = 0

    def get_sink_node_ids(self):
        return list(self._sink_ids)

    def get_from_sink(self, sink_node_id):
        frame = self._frames_by_sink.pop(sink_node_id, None)
        if not self._frames_by_sink:
            self.running = False
        return frame

    def get(self):
        self.get_calls += 1
        raise AssertionError("headless graph mode should not call get()")

    def get_audio(self):
        return None, None

    def stop(self):
        self.running = False


class _CountingRecorder(HeadlessMediaSink):
    def __init__(self):
        self.is_recording = True
        self.video_calls = 0
        self.write_calls = 0
        self.file_path = None

    def write_frame(self, video_frame) -> None:
        self.write_calls += 1

    def on_video_frame(self, video_frame) -> None:
        self.video_calls += 1

    def on_audio_chunk(self, audio_tensor, sample_rate) -> None:
        return

    def close(self) -> None:
        self.is_recording = False


class _FailingRecorder(_CountingRecorder):
    def on_video_frame(self, video_frame) -> None:
        raise RuntimeError("synthetic sink failure")


class _CollectingSink(HeadlessMediaSink):
    def __init__(self):
        self.frames = []

    def on_video_frame(self, video_frame) -> None:
        self.frames.append(torch.from_numpy(video_frame.to_ndarray(format="rgb24")))

    def on_audio_chunk(self, audio_tensor, sample_rate) -> None:
        return

    def close(self) -> None:
        return


@pytest.mark.anyio
async def test_headless_recorder_receives_primary_frame_once():
    frame = torch.zeros((16, 16, 3), dtype=torch.uint8)
    frame_processor = _SingleFrameProcessor(frame)
    session = HeadlessSession(frame_processor=frame_processor)

    recorder = _CountingRecorder()
    session._recorder = recorder
    session.add_media_sink(recorder)
    session._frame_consumer_running = True

    await session._consume_frames()

    assert recorder.video_calls == 1
    assert recorder.write_calls == 0


@pytest.mark.anyio
async def test_headless_clears_recorder_reference_on_sink_failure():
    frame = torch.zeros((16, 16, 3), dtype=torch.uint8)
    frame_processor = _SingleFrameProcessor(frame)
    session = HeadlessSession(frame_processor=frame_processor)

    recorder = _FailingRecorder()
    session._recorder = recorder
    session.add_media_sink(recorder)
    session._frame_consumer_running = True

    await session._consume_frames()

    assert session._recorder is None
    assert recorder not in session._get_sinks_snapshot()


@pytest.mark.anyio
async def test_headless_graph_mode_avoids_primary_queue_double_drain():
    primary = torch.full((16, 16, 3), 7, dtype=torch.uint8)
    frame_processor = _SinkOnlyFrameProcessor({"output": primary})
    session = HeadlessSession(frame_processor=frame_processor)

    sink = _CollectingSink()
    session.add_media_sink(sink)
    session._frame_consumer_running = True

    await session._consume_frames()

    assert frame_processor.get_calls == 0
    assert len(sink.frames) == 1
    assert torch.equal(sink.frames[0], primary)
    assert torch.equal(
        torch.from_numpy(session.get_last_frame("output").to_ndarray(format="rgb24")),
        primary,
    )


@pytest.mark.anyio
async def test_headless_graph_mode_keeps_latest_consumed_frame_behavior():
    primary = torch.full((16, 16, 3), 11, dtype=torch.uint8)
    secondary = torch.full((16, 16, 3), 22, dtype=torch.uint8)
    frame_processor = _SinkOnlyFrameProcessor(
        {"output": primary, "output_1": secondary}
    )
    session = HeadlessSession(frame_processor=frame_processor)

    sink = _CollectingSink()
    session.add_media_sink(sink)
    session._frame_consumer_running = True

    await session._consume_frames()

    assert frame_processor.get_calls == 0
    assert len(sink.frames) == 1
    assert torch.equal(sink.frames[0], primary)
    assert torch.equal(
        torch.from_numpy(session.get_last_frame("output").to_ndarray(format="rgb24")),
        primary,
    )
    assert torch.equal(
        torch.from_numpy(session.get_last_frame("output_1").to_ndarray(format="rgb24")),
        secondary,
    )
    assert torch.equal(
        torch.from_numpy(session.get_last_frame().to_ndarray(format="rgb24")),
        secondary,
    )
