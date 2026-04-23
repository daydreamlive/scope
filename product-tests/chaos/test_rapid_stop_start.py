"""Chaos — rapid Stop/Run toggling after a successful first frame.

Simulates the user who can't decide if they like what they see: for N
seconds, every 500–2000ms, click Stop then Run again. Asserts that no
click produces a retry, no session closes unexpectedly, and every Run
produces a new frame within a generous timeout.

This is exactly the pattern that exposes failure mode #2 — Scope-server ↔
remote-inference bad interactions when a session is torn down and brought
back up quickly.
"""

from __future__ import annotations

import pytest
from harness import flows, gates
from harness.chaos import ChaosDriver
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.chaos
def test_rapid_stop_start_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
    chaos_seed: str,
    test_report_dir,
):
    """Onboard, Run, hammer Stop/Run for 30s; every Run must land a frame."""
    report.metadata["workflow"] = "local-passthrough"
    report.metadata["chaos_seed"] = chaos_seed

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    toggles = {"count": 0, "frames_after_run": 0}

    def toggle_stop_start():
        failure_watcher.mark_initiated_stop()
        driver.click_testid("stream-run-stop")  # stop
        driver.page.wait_for_timeout(200)
        driver.click_testid("stream-run-stop")  # run
        try:
            driver.wait_first_frame(timeout_ms=20_000)
            toggles["frames_after_run"] += 1
        except Exception:
            pass
        toggles["count"] += 1

    chaos = ChaosDriver(seed=chaos_seed, report_dir=test_report_dir)
    chaos.register("toggle_stop_start", weight=1.0, fn=toggle_stop_start)
    chaos.run(duration_sec=30.0)

    report.measure("toggle_count", toggles["count"])
    report.measure("frames_landed_after_run", toggles["frames_after_run"])
    if toggles["count"] > 0 and toggles["frames_after_run"] < toggles["count"]:
        report.fail(
            f"only {toggles['frames_after_run']}/{toggles['count']} Run clicks produced a frame"
        )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
