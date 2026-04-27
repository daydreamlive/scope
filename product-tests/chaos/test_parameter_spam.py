"""Chaos — parameter spam during an active stream.

Simulates the kind of user who drags a slider at 60 Hz, or a programmatic
timeline that fires parameter updates as fast as it can. The session must
absorb the spam without retries, dropped frames, or UI error toasts.

Exposes failure mode where the parameters data channel back-pressures,
the frame processor gets behind, or a throttle elsewhere misfires and
kills the session.
"""

from __future__ import annotations

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
@pytest.mark.params
def test_parameter_spam_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
    chaos_seed: str,
    test_report_dir,
):
    """Onboard, Run, spam parameters for 30s, assert zero retries."""
    report.metadata["workflow"] = "local-passthrough"
    report.metadata["chaos_seed"] = chaos_seed

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    spam_counter = {"sent": 0, "failed": 0, "latency_ms": []}

    def spam_param():
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{scope_harness.base_url}/api/v1/session/parameters",
                json={"spam_key": f"v-{spam_counter['sent']}"},
                timeout=2.0,
            )
            r.raise_for_status()
            spam_counter["latency_ms"].append(int((time.perf_counter() - t0) * 1000))
        except Exception:
            spam_counter["failed"] += 1
        spam_counter["sent"] += 1

    chaos = ChaosDriver(
        seed=chaos_seed,
        report_dir=test_report_dir,
        tick_min_ms=20,
        tick_max_ms=80,
    )
    chaos.register("spam_param", weight=1.0, fn=spam_param)
    chaos.run(duration_sec=30.0)

    report.measure("spam_sent", spam_counter["sent"])
    report.measure("spam_failed", spam_counter["failed"])
    if spam_counter["latency_ms"]:
        sorted_lat = sorted(spam_counter["latency_ms"])
        p95 = sorted_lat[int(0.95 * (len(sorted_lat) - 1))]
        report.measure("spam_latency_ms_p95", p95)

    if spam_counter["failed"] > 0:
        report.fail(f"parameter apply failed {spam_counter['failed']}x during spam")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
