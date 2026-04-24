"""Stop-restart scenario — a user hits Stop, then Run again, and the session
must recover without retries or error toasts.

This is the least-exotic failure mode to regress: the backend holds a stale
session, or the frontend's WebRTC peer connection doesn't teardown, or the
Livepeer side reports an orphan. The scenario proves the happy cycle works.
"""

from __future__ import annotations

import pytest
from harness import baselines, flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.lifecycle
def test_stop_restart_local_passthrough(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Run → frame → Stop → Run → frame again. Two cycles, no retries."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")

    # Cycle 1
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=90_000)
    baselines.check(
        report, "local", "passthrough", "first_frame_time_ms_cycle1", int(first_ms)
    )
    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    # Cycle 2 — tests that Stop cleaned up so Run works again.
    second_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=90_000)
    baselines.check(
        report, "local", "passthrough", "first_frame_time_ms_cycle2", int(second_ms)
    )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
