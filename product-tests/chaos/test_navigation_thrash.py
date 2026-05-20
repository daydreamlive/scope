"""Chaos — user hits reload / back / forward during an active stream.

The browser fires ``beforeunload``, the frontend tears down the peer
connection, then the user lands right back on the app and expects to
resume. We do this 3 times in a row and assert the final state is a
streaming session with zero retry counters ticked.

This exposes two common bugs:
  - The frontend holds on to a stale PeerConnection after reload and
    refuses to bring up a new one.
  - The backend never hears the WebRTC close and thinks the old session
    is still live, producing ``session already active`` errors when the
    reloaded page tries to start.
"""

from __future__ import annotations

import pytest
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

pytestmark = pytest.mark.slow  # budget: ~3min; nightly-only


@pytest.mark.chaos
@pytest.mark.lifecycle
def test_navigation_thrash_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Run the app, thrash reload 3x while streaming; must recover every time."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    RELOAD_COUNT = 3
    recovery_ms: list[int] = []

    for i in range(RELOAD_COUNT):
        # Each reload tears down the PeerConnection. That's the normal user
        # path, so mark it initiated — otherwise the FailureWatcher will
        # (correctly) catch the close and blame us.
        failure_watcher.mark_initiated_stop()
        driver.page.reload(wait_until="domcontentloaded", timeout=30_000)

        # Onboarding state persists in DAYDREAM_SCOPE_DIR, so reload lands
        # directly on the graph view. Wait for Run to appear, then click it.
        driver.wait_testid("stream-run-stop", timeout_ms=30_000)
        elapsed = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
        recovery_ms.append(int(elapsed))
        report.metadata[f"recovery_{i}_ms"] = int(elapsed)

    sorted_rec = sorted(recovery_ms)
    p95 = sorted_rec[int(0.95 * (len(sorted_rec) - 1))]
    report.measure("navigation_thrash_recovery_ms_p95", p95)

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
