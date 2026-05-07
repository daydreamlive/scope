"""Tests for cloud-side pipeline_error forwarding.

Cloud architecture in scope:
- Cloud-side scope's PipelineProcessor emits a pipeline_error notification
  via its notification_callback (wired to _enqueue_notification on the
  cloud).
- Cloud's _forward_notifications_to_events writes
  {"type": "notification", "payload": <pipeline_error>} to the events
  channel.
- Local relay's livepeer_client._forward_runner_notification receives that
  payload and calls webrtc_manager.broadcast_notification(payload).
- broadcast_notification fans the payload out to every active browser
  session and, on a fatal error, schedules a defensive local-side
  FrameProcessor stop so the relay doesn't linger if the browser is slow
  to react.

These tests cover the local-relay side of that chain — the cloud side is
already covered by test_pipeline_error_notification.py.
"""

import time
import types

import pytest

from scope.server.livepeer_client import _forward_runner_notification
from scope.server.webrtc import WebRTCManager


class _FakeSender:
    def __init__(self):
        self.messages: list[dict] = []

    def call(self, message: dict) -> None:
        self.messages.append(message)


class _FakeFrameProcessor:
    def __init__(self):
        self.running = True
        self.stop_calls = 0

    def stop(self, error_message: str | None = None) -> None:
        # Mimic FrameProcessor.stop early-exit guard.
        if not self.running:
            return
        self.running = False
        self.stop_calls += 1


class _FakePeerConnection:
    def __init__(self, state: str = "connected"):
        self.connectionState = state


class _FakeSession:
    def __init__(self, sid: str):
        self.id = sid
        self.pc = _FakePeerConnection("connected")
        self.notification_sender = _FakeSender()
        self.frame_processor = _FakeFrameProcessor()


@pytest.fixture
def manager_with_session():
    manager = WebRTCManager()
    session = _FakeSession("session-1")
    manager.sessions[session.id] = session
    return manager, session


def test_pipeline_error_is_broadcast_to_all_sessions(manager_with_session):
    """A non-fatal pipeline_error from cloud reaches every active session."""
    manager, session = manager_with_session

    payload = {
        "type": "pipeline_error",
        "pipeline_id": "longlive",
        "node_id": "longlive",
        "message": "RuntimeError: boom",
        "exception_type": "RuntimeError",
        "fatal": False,
        "recoverable": True,
    }

    manager.broadcast_notification(payload)

    assert session.notification_sender.messages == [payload]
    # Non-fatal: do NOT stop the relay's frame processor.
    assert session.frame_processor.running is True
    assert session.frame_processor.stop_calls == 0


def test_fatal_pipeline_error_also_stops_local_frame_processor(manager_with_session):
    """A fatal pipeline_error tears down the local relay too."""
    manager, session = manager_with_session

    payload = {
        "type": "pipeline_error",
        "pipeline_id": "longlive",
        "node_id": "longlive",
        "message": "CUDA OOM",
        "exception_type": "OutOfMemoryError",
        "fatal": True,
        "recoverable": False,
    }

    manager.broadcast_notification(payload)

    # Browser still gets the message.
    assert session.notification_sender.messages == [payload]

    # Defensive stop runs on a daemon thread; wait briefly.
    deadline = time.time() + 2.0
    while time.time() < deadline and session.frame_processor.running:
        time.sleep(0.02)

    assert session.frame_processor.running is False
    assert session.frame_processor.stop_calls == 1


def test_stream_stopped_with_error_also_tears_down(manager_with_session):
    """stream_stopped carrying fatal=True tears down the local relay too."""
    manager, session = manager_with_session

    payload = {
        "type": "stream_stopped",
        "error_message": "Pipeline died",
        "fatal": True,
    }

    manager.broadcast_notification(payload)

    deadline = time.time() + 2.0
    while time.time() < deadline and session.frame_processor.running:
        time.sleep(0.02)

    assert session.frame_processor.running is False


def test_stream_stopped_without_error_does_not_tear_down(manager_with_session):
    """A normal stream_stopped (e.g. user clicked Stop) does NOT tear down."""
    manager, session = manager_with_session

    manager.broadcast_notification({"type": "stream_stopped"})

    # Give any (incorrect) thread a chance to run.
    time.sleep(0.1)
    assert session.frame_processor.running is True
    assert session.frame_processor.stop_calls == 0


def test_closed_session_is_skipped(manager_with_session):
    """Closed/failed sessions don't receive notifications."""
    manager, session = manager_with_session
    session.pc.connectionState = "closed"

    manager.broadcast_notification(
        {"type": "pipeline_error", "fatal": True, "message": "x"}
    )

    assert session.notification_sender.messages == []
    assert session.frame_processor.running is True


def test_forward_runner_notification_routes_to_broadcast():
    """_forward_runner_notification looks up webrtc_manager and forwards."""
    fake_manager = types.SimpleNamespace(received=[])
    fake_manager.broadcast_notification = lambda payload: fake_manager.received.append(
        payload
    )

    payload = {"type": "pipeline_error", "fatal": True, "message": "x"}

    import scope.server.app as real_app

    _missing = object()
    saved = getattr(real_app, "webrtc_manager", _missing)
    try:
        real_app.webrtc_manager = fake_manager
        _forward_runner_notification(payload)
    finally:
        if saved is _missing:
            delattr(real_app, "webrtc_manager")
        else:
            real_app.webrtc_manager = saved

    assert fake_manager.received == [payload]


def test_forward_runner_notification_ignores_non_dict():
    """Garbage payloads from the events channel are dropped silently."""
    # Should not raise even if webrtc_manager is missing.
    _forward_runner_notification("not a dict")
    _forward_runner_notification(None)
    _forward_runner_notification(42)
