"""Parameter-apply scenario — parameters actually land, fast, and round-trip.

Covers the silent-failure mode where a slider looks like it moved in the UI
but the backend never applied the change. We start a session, POST a
parameter, GET it back, and assert:

  - the applied value matches what we sent
  - the round-trip fits within the SLO ceiling
  - no retries, no unexpected closes, no error toasts

This uses ``local-passthrough`` on CPU because the test proves the wiring
(HTTP → WebRTC data channel → frame processor → broadcast), not a specific
pipeline's response to parameters.
"""

from __future__ import annotations

import time

import requests
from harness import baselines, flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


def _apply_and_readback(base_url: str, params: dict) -> tuple[dict, int]:
    """POST params, then GET — return (readback, round_trip_ms)."""
    t0 = time.perf_counter()
    r = requests.post(f"{base_url}/api/v1/session/parameters", json=params, timeout=5.0)
    r.raise_for_status()
    g = requests.get(f"{base_url}/api/v1/session/parameters", timeout=5.0)
    g.raise_for_status()
    rt = int((time.perf_counter() - t0) * 1000)
    return g.json().get("parameters", {}), rt


def test_parameter_apply_local_passthrough(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Start a stream, change a parameter, assert round-trip."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    flows.start_stream_and_wait_first_frame(driver, timeout_ms=90_000)

    # The schema allows arbitrary extra keys; the passthrough pipeline ignores
    # them but the endpoint still round-trips through the frame processor and
    # broadcasts a parameters_updated event to any connected clients.
    test_value = f"pt-{int(time.time())}"
    params = {"test_key": test_value, "prompt_interpolation_method": "linear"}

    round_trips: list[int] = []
    for _ in range(5):
        readback, rt_ms = _apply_and_readback(scope_harness.base_url, params)
        round_trips.append(rt_ms)
        assert readback.get("test_key") == test_value, (
            f"parameter did not round-trip: sent={params} got={readback}"
        )

    p95 = sorted(round_trips)[int(0.95 * (len(round_trips) - 1))]
    baselines.check(report, "local", "passthrough", "parameter_round_trip_ms_p95", p95)

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
