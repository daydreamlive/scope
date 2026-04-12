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
