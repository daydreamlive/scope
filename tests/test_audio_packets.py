import queue
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch
from av import AudioFrame

from scope.server.cloud_relay import CloudRelay
from scope.server.frame_processor import FrameProcessor
from scope.server.media_packets import AudioPacket, MediaTimestamp


def _make_frame_processor_with_audio_queue(items):
    processor = object.__new__(FrameProcessor)
    processor.running = True
    processor._cloud_relay = None
    audio_queue = queue.Queue()
    for item in items:
        audio_queue.put_nowait(item)
    processor._sink_processor = SimpleNamespace(audio_output_queue=audio_queue)
    return processor


def test_get_audio_packet_accepts_legacy_audio_tuple():
    audio = torch.ones((2, 8))
    processor = _make_frame_processor_with_audio_queue([(audio, 48_000)])

    packet = processor.get_audio_packet()

    assert packet == AudioPacket(audio=audio, sample_rate=48_000)


def test_get_audio_remains_backward_compatible_for_audio_packets():
    audio = torch.zeros((1, 16))
    packet = AudioPacket(
        audio=audio,
        sample_rate=24_000,
        timestamp=MediaTimestamp(pts=7),
    )
    processor = _make_frame_processor_with_audio_queue([packet])

    assert processor.get_audio() == (audio, 24_000)


def test_get_audio_packet_accepts_legacy_cloud_tuple():
    audio = torch.randn((2, 16))
    processor = object.__new__(FrameProcessor)
    processor.running = True
    processor._sink_processor = None
    processor._cloud_relay = SimpleNamespace(get_audio=lambda: (audio, 48_000))

    packet = processor.get_audio_packet()

    assert packet == AudioPacket(audio=audio, sample_rate=48_000)


def test_cloud_relay_audio_packet_preserves_timestamp():
    class _FakeCloudManager:
        def add_frame_callback(self, callback):  # pragma: no cover - unused in test
            return None

        def add_audio_callback(self, callback):  # pragma: no cover - unused in test
            return None

        def remove_frame_callback(self, callback):  # pragma: no cover - unused in test
            return None

        def remove_audio_callback(self, callback):  # pragma: no cover - unused in test
            return None

    relay = CloudRelay(_FakeCloudManager())
    audio_np = np.zeros((2, 160), dtype=np.float32)
    frame = AudioFrame.from_ndarray(audio_np, format="fltp", layout="stereo")
    frame.sample_rate = 48_000
    frame.pts = 321
    frame.time_base = Fraction(1, 48_000)

    relay.on_audio_from_cloud(frame)
    packet = relay.get_audio()

    assert packet is not None
    assert packet.sample_rate == 48_000
    assert packet.timestamp == MediaTimestamp(pts=321, time_base=Fraction(1, 48_000))
