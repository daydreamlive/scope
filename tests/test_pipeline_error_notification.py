"""Tests for pipeline error notification (user-facing error UX).

Verifies that when a pipeline raises errors during streaming, the
PipelineProcessor:
  1. Emits a `pipeline_error` message via the notification callback so the
     frontend can render a toast / inline overlay (no more blank screen).
  2. Invokes the on_fatal_error callback so the FrameProcessor can stop the
     stream cleanly.
  3. Escalates a stuck pipeline (>= MAX_CONSECUTIVE_RECOVERABLE_ERRORS) to
     fatal even when the underlying exceptions are technically "recoverable".

No GPU required; uses a stub pipeline.
"""

import queue
import time

import pytest
import torch

from scope.core.nodes.base import NodeDefinition
from scope.server.pipeline_processor import (
    MAX_CONSECUTIVE_RECOVERABLE_ERRORS,
    PipelineProcessor,
)


class _AlwaysFailsPipeline:
    """Stub pipeline that raises a recoverable error on every call."""

    def __init__(self, exc=RuntimeError("boom")):
        self._exc = exc
        self.call_count = 0

    def get_definition(self):
        return NodeDefinition(node_type_id="failing_stub", display_name="FailingStub")

    def prepare(self, **_):
        return None

    def __call__(self, **_):
        self.call_count += 1
        raise self._exc


class _OOMPipeline:
    """Stub pipeline that raises CUDA OOM (non-recoverable) on every call."""

    def get_definition(self):
        return NodeDefinition(node_type_id="oom_stub", display_name="OOMStub")

    def prepare(self, **_):
        return None

    def __call__(self, **_):
        raise torch.cuda.OutOfMemoryError("simulated OOM")


def _wait_for(predicate, *, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_recoverable_errors_escalate_to_fatal_and_notify():
    """A pipeline that fails repeatedly should emit pipeline_error and stop."""
    notifications: list[dict] = []
    fatal_calls: list[tuple[str, str, str]] = []

    pipeline = _AlwaysFailsPipeline()
    processor = PipelineProcessor(
        pipeline=pipeline,
        pipeline_id="failing-pipeline",
        node_id="node-A",
        notification_callback=notifications.append,
        on_fatal_error=lambda pid, nid, msg: fatal_calls.append((pid, nid, msg)),
    )
    processor.output_queues["video"] = [queue.Queue(maxsize=4)]
    processor.start()

    try:
        # Worker should escalate after MAX_CONSECUTIVE_RECOVERABLE_ERRORS errors.
        assert _wait_for(lambda: len(fatal_calls) >= 1, timeout=10.0), (
            "Expected on_fatal_error to be invoked after consecutive failures, "
            f"but only saw {len(fatal_calls)} fatal calls and "
            f"{pipeline.call_count} pipeline calls"
        )
    finally:
        processor.stop()

    # The fatal callback received the right pipeline / node identifiers.
    pid, nid, msg = fatal_calls[0]
    assert pid == "failing-pipeline"
    assert nid == "node-A"
    assert "RuntimeError" in msg or "boom" in msg

    # A pipeline_error notification with fatal=True was sent.
    fatal_msgs = [
        m
        for m in notifications
        if m.get("type") == "pipeline_error" and m.get("fatal") is True
    ]
    assert fatal_msgs, (
        f"No fatal pipeline_error notification emitted. "
        f"Received: {[m.get('type') for m in notifications]}"
    )
    msg = fatal_msgs[0]
    assert msg["pipeline_id"] == "failing-pipeline"
    assert msg["node_id"] == "node-A"
    assert msg["recoverable"] is False
    assert "boom" in msg["message"] or "RuntimeError" in msg["message"]

    # We made roughly MAX_CONSECUTIVE_RECOVERABLE_ERRORS attempts before stopping.
    assert pipeline.call_count >= MAX_CONSECUTIVE_RECOVERABLE_ERRORS


def test_non_recoverable_error_notifies_and_stops_immediately():
    """CUDA OOM is non-recoverable: should emit fatal pipeline_error after 1 call."""
    pytest.importorskip("torch")
    if not hasattr(torch.cuda, "OutOfMemoryError"):
        pytest.skip("torch version too old for OutOfMemoryError")

    notifications: list[dict] = []
    fatal_calls: list[tuple[str, str, str]] = []

    processor = PipelineProcessor(
        pipeline=_OOMPipeline(),
        pipeline_id="oom-pipeline",
        node_id="node-B",
        notification_callback=notifications.append,
        on_fatal_error=lambda pid, nid, msg: fatal_calls.append((pid, nid, msg)),
    )
    processor.output_queues["video"] = [queue.Queue(maxsize=4)]
    processor.start()

    try:
        assert _wait_for(lambda: len(fatal_calls) >= 1, timeout=5.0)
    finally:
        processor.stop()

    fatal_msgs = [
        m
        for m in notifications
        if m.get("type") == "pipeline_error" and m.get("fatal") is True
    ]
    assert fatal_msgs, "Expected a fatal pipeline_error for non-recoverable exception"
    assert "OutOfMemoryError" in fatal_msgs[0]["exception_type"]
    assert fatal_msgs[0]["recoverable"] is False
