"""Chaos — /session/start called twice without a /stop in between.

Double-click, duplicate tab, timeline auto-run firing on top of a user
click — whatever the path, ``POST /api/v1/session/start`` can race with
itself. The server must pick a winner without crashing and without
closing the already-running session.

Acceptable outcomes for the second call:
  - 2xx (server treats idempotently and returns the same active session)
  - 4xx (server rejects: "session already active")

Not acceptable:
  - 5xx / process crash
  - First session forcibly closed (unexpected_close counter ticks)
  - Banned retry counters tick
"""

from __future__ import annotations

import concurrent.futures as cf

import pytest
import requests
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


@pytest.mark.chaos
@pytest.mark.lifecycle
def test_double_start_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Start via UI, then fire a second HTTP start in parallel. Expect sanity."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    base = scope_harness.base_url
    # Use a passthrough-shaped start body. If the server rejects it for any
    # other reason (schema mismatch) we'll see it in the status code and
    # the test still measures the blast radius.
    body = {
        "pipeline_id": "passthrough",
        "input_mode": "camera",
    }

    def attempt_start():
        try:
            r = requests.post(f"{base}/api/v1/session/start", json=body, timeout=5.0)
            return r.status_code, r.text[:200]
        except Exception as e:
            return -1, f"{type(e).__name__}:{e}"

    # Fire 3 near-simultaneous /start calls.
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(attempt_start) for _ in range(3)]
        results = [f.result(timeout=15) for f in futures]

    report.metadata["double_start_results"] = [
        {"status": s, "body": b[:80]} for s, b in results
    ]
    crashes = [r for r in results if r[0] == -1 or r[0] >= 500]
    if crashes:
        report.fail(f"double-start produced {len(crashes)} 5xx/timeout: {crashes}")

    # The original stream MUST still be alive — no forced close.
    try:
        driver.wait_first_frame(timeout_ms=10_000)
    except Exception:
        report.fail("original stream no longer producing frames after double-start")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
