"""Chaos — reload the page mid-stream; session must clean up and not leak.

"Input switching" in the user-observable sense: the user navigates away,
refreshes the tab, or otherwise tears down the frontend without clicking
Stop. The backend session must recognize the disconnect cleanly — no
zombie session, no spurious reconnect, no unexpected_session_close event
because the test-initiated page unload is expected.

A well-behaved Scope emits an expected WebRTC close and starts a fresh
session after reload. The test asserts:
  - After reload, onboarding does NOT re-appear (the user already
    completed it, and state persists in the isolated DAYDREAM_SCOPE_DIR).
  - Clicking Run again produces a new first frame.
  - No retry counters incremented.
"""

from __future__ import annotations

import pytest
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.chaos
def test_reload_mid_stream_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Onboard → Run → reload → Run again. Both Runs must produce a frame."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms_before_reload", int(first_ms))

    # Reload — the frontend tears down but the backend should clean the session.
    failure_watcher.mark_initiated_stop()
    driver.page.reload(wait_until="domcontentloaded")

    # The app should land directly on the graph view, Run button visible.
    # (Onboarding completion is sticky in DAYDREAM_SCOPE_DIR.)
    driver.wait_testid("stream-run-stop", timeout_ms=30_000)

    second_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms_after_reload", int(second_ms))

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
