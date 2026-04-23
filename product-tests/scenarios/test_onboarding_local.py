"""Onboarding smoke — local inference, passthrough workflow, first frame.

The "if this is red, ship nothing" gate. Drives real UI through Playwright
against a real Scope subprocess and asserts:
  - Onboarding completes without a single error toast
  - Stream starts and a video frame renders
  - RetryProbe sees zero retry/drop events at teardown
  - FailureWatcher sees zero unexpected session closes

This scenario uses the `local-passthrough` starter workflow so it runs
CPU-only and fits within the PR gate's 25-minute budget.
"""

from __future__ import annotations

from harness import baselines, flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


def test_onboarding_local_passthrough(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Cold-start → pick local → decline telemetry → pick Camera Preview → Run → first frame."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")

    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=90_000)
    baselines.check(
        report, "local", "passthrough", "first_frame_time_ms", int(first_ms)
    )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    # Clean stop so the autouse watcher doesn't see a stray close.
    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
