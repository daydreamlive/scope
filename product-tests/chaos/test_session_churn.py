"""Chaos — combined session churn for a long duration.

This is the test that runs nightly as the "did we break anything?"
canary: a longer-running sample of all the individual chaos actions
combined, with a seeded RNG so runs are reproducible.

Mixes:
  - Stop/Run toggles (most common user action)
  - Parameter spam (slider drag)
  - Reload (tab refresh)

If any individual chaos test passes but this combined one fails, the
bug is almost certainly in how two of the chaos actions interact —
often a race between parameter apply and session teardown, or a stale
reconnect fired from the frontend after a reload.

Duration defaults to 60s on the PR gate and can be overridden via
SCOPE_CHURN_DURATION_SEC for nightly.
"""

from __future__ import annotations

import os
import time

import pytest
import requests
from harness import flows, gates
from harness.chaos import ChaosDriver
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.chaos
@pytest.mark.slow
@pytest.mark.lifecycle
def test_session_churn_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
    chaos_seed: str,
    test_report_dir,
):
    """60s of combined stop/start + parameter spam + reload churn."""
    report.metadata["workflow"] = "local-passthrough"
    report.metadata["chaos_seed"] = chaos_seed

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)

    counters = {
        "toggle": 0,
        "param": 0,
        "reload": 0,
        "errors": 0,
    }

    def toggle_stop_start():
        try:
            failure_watcher.mark_initiated_stop()
            driver.click_testid("stream-run-stop")
            driver.page.wait_for_timeout(150)
            driver.click_testid("stream-run-stop")
            counters["toggle"] += 1
        except Exception:
            counters["errors"] += 1

    def spam_param():
        try:
            r = requests.post(
                f"{scope_harness.base_url}/api/v1/session/parameters",
                json={"churn_key": str(time.time_ns())},
                timeout=2.0,
            )
            r.raise_for_status()
            counters["param"] += 1
        except Exception:
            counters["errors"] += 1

    def reload_page():
        try:
            failure_watcher.mark_initiated_stop()
            driver.page.reload(wait_until="domcontentloaded")
            driver.wait_testid("stream-run-stop", timeout_ms=30_000)
            driver.click_testid("stream-run-stop")
            counters["reload"] += 1
        except Exception:
            counters["errors"] += 1

    duration = float(os.environ.get("SCOPE_CHURN_DURATION_SEC", "60"))

    chaos = ChaosDriver(
        seed=chaos_seed,
        report_dir=test_report_dir,
        tick_min_ms=250,
        tick_max_ms=1500,
    )
    chaos.register("toggle_stop_start", weight=5.0, fn=toggle_stop_start)
    chaos.register("spam_param", weight=3.0, fn=spam_param)
    chaos.register("reload_page", weight=1.0, fn=reload_page)
    chaos.run(duration_sec=duration)

    for k, v in counters.items():
        report.measure(f"churn_{k}", v)

    if counters["errors"] > 0:
        report.fail(f"chaos actions failed: {counters['errors']}x")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    # Best-effort cleanup.
    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
