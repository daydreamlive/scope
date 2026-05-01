"""Tests for pipeline frame delivery smoothness.

Uses a stub pipeline with the real PipelineProcessor to verify that
frame delivery is smooth across different batch sizes, processing times,
and pause/resume scenarios. No GPU required.
"""

import queue
import random
import statistics
import time

import torch

from scope.core.nodes.base import NodeDefinition
from scope.server.pipeline_processor import PipelineProcessor


class StubPipeline:
    """Minimal pipeline with controllable batch size and processing delay."""

    def __init__(self, batch_size: int = 12, delay: float = 0.5, variance: float = 0.0):
        self.batch_size = batch_size
        self.delay = delay
        self.variance = variance
        self.call_count = 0

    def get_definition(self):
        return NodeDefinition(node_type_id="stub", display_name="Stub")

    def prepare(self, **kwargs):
        return None

    def __call__(self, **kwargs):
        self.call_count += 1
        actual_delay = self.delay
        if self.variance > 0:
            actual_delay = max(0.01, self.delay + random.gauss(0, self.variance))
        time.sleep(actual_delay)
        return {"video": torch.rand(self.batch_size, 64, 64, 3)}


def _measure_delivery(
    batch_size: int,
    delay: float,
    duration: float,
    variance: float = 0.0,
    pause_at: float | None = None,
    pause_duration: float = 2.0,
) -> dict:
    """Run a pipeline processor and measure consumer-side frame delivery timing.

    Simulates the WebRTC recv() pacing behavior: reads frames from the output
    queue and sleeps for frame_ptime between deliveries, matching the real
    consumer in tracks.py.
    """
    pipeline = StubPipeline(batch_size=batch_size, delay=delay, variance=variance)
    processor = PipelineProcessor(
        pipeline=pipeline,
        pipeline_id="test-stub",
    )
    processor.output_queues["video"] = [queue.Queue(maxsize=30)]
    processor.start()

    intervals = []
    fps_readings = []
    last_delivery = None
    frames_received = 0
    empty_polls = 0
    pause_handled = False
    ideal_fps = batch_size / delay

    start = time.time()
    deadline = start + duration + delay * 3

    while time.time() < deadline:
        elapsed = time.time() - start
        if elapsed > duration:
            break

        if pause_at is not None and not pause_handled and elapsed >= pause_at:
            processor.update_parameters({"paused": True})
            time.sleep(pause_duration)
            processor.update_parameters({"paused": False})
            last_delivery = None
            pause_handled = True

        fps = processor.get_fps()
        frame_ptime = 1.0 / fps

        try:
            processor.output_queue.get_nowait()
        except queue.Empty:
            if last_delivery is not None:
                empty_polls += 1
            time.sleep(0.01)
            continue

        now = time.time()

        if last_delivery is not None:
            time_since_last = now - last_delivery
            wait = frame_ptime - time_since_last
            if wait > 0:
                time.sleep(wait)

            actual_interval = time.time() - last_delivery
            intervals.append(actual_interval)
            fps_readings.append(fps)

        last_delivery = time.time()
        frames_received += 1

    processor.stop()

    ideal_interval = 1.0 / ideal_fps
    mean_interval = statistics.mean(intervals)
    stdev_interval = statistics.stdev(intervals)
    jitter_pct = (stdev_interval / mean_interval) * 100
    stall_threshold = ideal_interval * 1.5
    stall_ratio = (
        sum(1 for i in intervals if i > stall_threshold) / len(intervals) * 100
    )
    fps_mean = statistics.mean(fps_readings)
    fps_stdev = statistics.stdev(fps_readings) if len(fps_readings) > 1 else 0
    fps_error_pct = abs(fps_mean - ideal_fps) / ideal_fps * 100

    return {
        "jitter_pct": jitter_pct,
        "stall_ratio_pct": stall_ratio,
        "fps_error_pct": fps_error_pct,
        "fps_stdev": fps_stdev,
        "max_interval_ms": max(intervals) * 1000,
        "ideal_interval_ms": ideal_interval * 1000,
        "empty_polls": empty_polls,
    }


class TestFrameDelivery:
    """Verify that frame delivery is smooth across pipeline configurations."""

    def test_large_batch_delivery(self):
        """LongLive-like: 12-frame batches should deliver smoothly."""
        random.seed(42)
        result = _measure_delivery(batch_size=12, delay=0.5, duration=10)

        assert result["jitter_pct"] < 10, (
            f"Jitter {result['jitter_pct']:.1f}% exceeds 10% threshold"
        )
        assert result["stall_ratio_pct"] < 1, (
            f"Stall ratio {result['stall_ratio_pct']:.1f}% exceeds 1% threshold"
        )
        assert result["fps_error_pct"] < 5, (
            f"FPS error {result['fps_error_pct']:.1f}% exceeds 5% threshold"
        )

    def test_small_batch_delivery(self):
        """SDv2-like: 4-frame batches with variance should deliver smoothly."""
        random.seed(42)
        result = _measure_delivery(batch_size=4, delay=0.2, duration=10, variance=0.03)

        assert result["jitter_pct"] < 15, (
            f"Jitter {result['jitter_pct']:.1f}% exceeds 15% threshold"
        )
        assert result["stall_ratio_pct"] < 2, (
            f"Stall ratio {result['stall_ratio_pct']:.1f}% exceeds 2% threshold"
        )
        assert result["fps_error_pct"] < 10, (
            f"FPS error {result['fps_error_pct']:.1f}% exceeds 10% threshold"
        )

    def test_single_frame_delivery(self):
        """Single-frame pipeline should deliver smoothly (no regression)."""
        random.seed(42)
        result = _measure_delivery(
            batch_size=1, delay=0.033, duration=10, variance=0.005
        )

        assert result["jitter_pct"] < 15, (
            f"Jitter {result['jitter_pct']:.1f}% exceeds 15% threshold"
        )
        assert result["stall_ratio_pct"] < 2, (
            f"Stall ratio {result['stall_ratio_pct']:.1f}% exceeds 2% threshold"
        )

    def test_pause_resume_delivery(self):
        """Frame delivery should recover smoothly after pause/resume."""
        random.seed(42)
        result = _measure_delivery(
            batch_size=12, delay=0.5, duration=15, pause_at=5.0, pause_duration=2.0
        )

        assert result["jitter_pct"] < 10, (
            f"Jitter {result['jitter_pct']:.1f}% exceeds 10% threshold after pause/resume"
        )
        assert result["stall_ratio_pct"] < 1, (
            f"Stall ratio {result['stall_ratio_pct']:.1f}% exceeds 1% threshold after pause/resume"
        )
        assert result["fps_error_pct"] < 5, (
            f"FPS error {result['fps_error_pct']:.1f}% exceeds 5% threshold after pause/resume"
        )
