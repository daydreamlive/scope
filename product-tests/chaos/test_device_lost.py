"""Chaos — camera/device yanked mid-session.

Simulates the user unplugging a USB webcam, macOS reassigning the camera
to another app, or the OS putting the device to sleep. We do this by
reaching into ``navigator.mediaDevices`` and calling ``stop()`` on every
active track, which fires ``ended`` events on the MediaStreamTrack —
exactly what the browser does on a real device loss.

The session must either:
  - Cleanly surface a user-facing error ("camera unavailable") and let
    the user recover, OR
  - Automatically reacquire and keep going.

What must NOT happen: silent freeze, retry counter tick, unexpected
session close, or a crash. The gate catches all four.
"""

from __future__ import annotations

import pytest
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

# Walk every global MediaStream / MediaStreamTrack and end it. We use a
# WeakSet-ish approach: intercept getUserMedia to capture every stream,
# then call stop() on all of them. Run at page load via addInitScript so
# it's already wired up before the first getUserMedia fires.
_TRACK_INTERCEPT = """() => {
    if (window.__capturedStreams) return;
    window.__capturedStreams = [];
    const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
        const s = await gum(constraints);
        window.__capturedStreams.push(s);
        return s;
    };
}"""

_END_ALL_TRACKS = """() => {
    const streams = window.__capturedStreams || [];
    let ended = 0;
    for (const s of streams) {
        for (const t of s.getTracks()) {
            if (t.readyState === 'live') {
                t.stop();
                // Also dispatch the 'ended' event explicitly — t.stop()
                // does not on all browsers, and listeners key off it.
                t.dispatchEvent(new Event('ended'));
                ended++;
            }
        }
    }
    return ended;
}"""


@pytest.mark.chaos
@pytest.mark.input
def test_device_lost_mid_stream_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Stream, kill all camera tracks, assert the system reacts sanely."""
    report.metadata["workflow"] = "local-passthrough"

    # Install the getUserMedia interceptor BEFORE onboarding so every
    # stream request is captured.
    driver.context.add_init_script(_TRACK_INTERCEPT)

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    # Kill every live track. Returns the count for sanity.
    tracks_ended = driver.page.evaluate(_END_ALL_TRACKS)
    report.measure("tracks_ended", int(tracks_ended or 0))

    # The stream either recovers or cleanly stops. Give it time; then
    # check both that the retry gates are clean and that any error
    # surfaced is a user-facing message, not a silent failure.
    driver.page.wait_for_timeout(5000)

    # If the stream auto-recovered, great. If it stopped, the Run button
    # should be back to an actionable state. Check: there's no infinite
    # spinner and no unhandled error toast that says "internal error".
    error_count = driver.error_toast_count()
    report.measure("error_toasts_after_device_loss", error_count)
    # A graceful user-facing message is allowed; an "internal" one is not.
    bad_messages = driver.page.locator(
        "text=/internal error|uncaught|undefined is not/i"
    ).count()
    if bad_messages > 0:
        report.fail("uncaught/internal error surfaced after device loss")

    # Server must still be healthy.
    import requests

    try:
        h = requests.get(f"{scope_harness.base_url}/health", timeout=3.0)
        assert h.status_code == 200
    except Exception as e:
        report.fail(f"server health check failed after device loss: {e}")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
