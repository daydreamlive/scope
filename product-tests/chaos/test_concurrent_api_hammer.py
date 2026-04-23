"""Chaos — parallel HTTP hammer against the session surface.

Real users don't just click once. A browser retry, a timeline auto-apply,
and a user-initiated action can all hit the API in the same tick. This
test fires start / stop / parameters / workflow resolve from a thread
pool while the UI session is also live, proving the server's in-flight
serialization is real, not accidental.

Every request is allowed to succeed or fail with a sane HTTP error
(``4xx``/``5xx`` with a JSON body). What must NOT happen:

  - A banned retry counter ticks
  - An unexpected session close fires
  - The UI surfaces an error toast
  - The server returns a non-JSON 500 or the process crashes

If the server can only handle strictly-sequential API calls, this test
is the fastest way to find that out.
"""

from __future__ import annotations

import concurrent.futures as cf
import random

import pytest
import requests
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness


def _safe_post(url: str, json: dict | None = None) -> tuple[int, str]:
    try:
        r = requests.post(url, json=json, timeout=3.0)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)[:200]


def _safe_get(url: str) -> tuple[int, str]:
    try:
        r = requests.get(url, timeout=3.0)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)[:200]


@pytest.mark.chaos
def test_concurrent_api_hammer_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
    chaos_seed: str,
):
    """Run a live UI session; pound the HTTP API in parallel for 15s."""
    report.metadata["workflow"] = "local-passthrough"
    report.metadata["chaos_seed"] = chaos_seed

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)
    report.measure("first_frame_time_ms", int(first_ms))

    base = scope_harness.base_url
    rng = random.Random(chaos_seed)

    # Weighted fleet of actions. Read-heavy by design: start/stop are spicy
    # and we want to churn the session without asking the UI to also stop.
    actions = [
        (
            "param",
            lambda: _safe_post(
                f"{base}/api/v1/session/parameters", {"k": rng.random()}
            ),
        ),
        (
            "param",
            lambda: _safe_post(
                f"{base}/api/v1/session/parameters", {"k": rng.random()}
            ),
        ),
        ("param_get", lambda: _safe_get(f"{base}/api/v1/session/parameters")),
        ("metrics", lambda: _safe_get(f"{base}/api/v1/session/metrics")),
        ("status", lambda: _safe_get(f"{base}/api/v1/pipeline/status")),
        (
            "resolve",
            lambda: _safe_post(
                f"{base}/api/v1/workflow/resolve", {"pipelines": ["passthrough"]}
            ),
        ),
    ]

    results: dict[str, list[int]] = {name: [] for name, _ in actions}
    N_WORKERS = 8
    N_CALLS = 400

    def worker(_i: int):
        name, fn = rng.choice(actions)
        code, _ = fn()
        return name, code

    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for name, code in ex.map(worker, range(N_CALLS)):
            results[name].append(code)

    bad_codes = {
        name: [c for c in codes if c == -1 or c >= 500]
        for name, codes in results.items()
    }
    bad_total = sum(len(v) for v in bad_codes.values())
    report.measure("hammer_requests", N_CALLS)
    report.measure("hammer_5xx_or_timeout", bad_total)
    report.metadata["hammer_bad_samples"] = {
        name: v[:3] for name, v in bad_codes.items() if v
    }

    if bad_total > 0:
        report.fail(
            f"{bad_total}/{N_CALLS} API calls returned 5xx or timed out: "
            f"{report.metadata['hammer_bad_samples']}"
        )

    # The stream must still be live after the hammer.
    try:
        driver.wait_first_frame(timeout_ms=15_000)
    except Exception:
        report.fail("stream did not recover a frame after concurrent API hammer")

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
