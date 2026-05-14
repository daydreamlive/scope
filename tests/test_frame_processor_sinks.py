import numpy as np
import torch

from scope.server.frame_processor import FrameProcessor
from scope.server.media_packets import VideoPacket


class _FakeSinkManager:
    def __init__(self, packets: dict[str, VideoPacket]):
        self._packets = dict(packets)
        self.fanned_frames: list[np.ndarray] = []

    @property
    def has_generic_sinks(self) -> bool:
        return True

    def get_packet_from_sink(self, sink_node_id: str) -> VideoPacket | None:
        return self._packets.pop(sink_node_id, None)

    def fan_out_frame(self, frame_np: np.ndarray) -> None:
        self.fanned_frames.append(frame_np.copy())


def _make_frame_processor(primary_sink_node_id: str) -> FrameProcessor:
    processor = object.__new__(FrameProcessor)
    processor.running = True
    processor._primary_sink_node_id = primary_sink_node_id
    processor._frames_out = 0
    processor._playback_ready_emitted = True
    return processor


def test_primary_graph_sink_feeds_generic_output_sinks():
    frame = torch.full((2, 3, 3), 127, dtype=torch.uint8)
    sink_manager = _FakeSinkManager({"output": VideoPacket(tensor=frame)})
    processor = _make_frame_processor("output")
    processor.sink_manager = sink_manager

    packet = processor.get_packet_from_sink("output")

    assert packet is not None
    assert processor._frames_out == 1
    assert len(sink_manager.fanned_frames) == 1
    np.testing.assert_array_equal(sink_manager.fanned_frames[0], frame.numpy())


def test_secondary_graph_sink_does_not_duplicate_generic_output_sinks():
    frame = torch.full((2, 3, 3), 255, dtype=torch.uint8)
    sink_manager = _FakeSinkManager({"secondary": VideoPacket(tensor=frame)})
    processor = _make_frame_processor("output")
    processor.sink_manager = sink_manager

    packet = processor.get_packet_from_sink("secondary")

    assert packet is not None
    assert processor._frames_out == 1
    assert sink_manager.fanned_frames == []
