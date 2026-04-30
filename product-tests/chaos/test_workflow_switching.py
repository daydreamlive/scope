"""Chaos — switch the active pipeline between two CPU pipelines.

The user opens a different workflow while one is already active. Internally
Scope must stop the current session, unload the old pipeline, load the new
one, and restart cleanly. This exposes races where the old session's
teardown overlaps the new session's setup.

On CPU-only rings we cycle between ``passthrough`` and ``gray`` — both
lightweight preprocessors that boot in under a second.
"""

from __future__ import annotations

import time

import pytest
import requests
from harness import flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

CPU_PIPELINES = ["passthrough", "gray"]


def _swap_pipeline(base_url: str, pipeline_id: str) -> None:
    """Stop current session, load new pipeline, start a fresh session.

    Uses the HTTP API because this chaos is about backend resilience to
    the transition; UI driving would bog the test down in graph editor
    interactions that aren't what's being tested.
    """
    requests.post(f"{base_url}/api/v1/session/stop", timeout=10.0)
    r = requests.post(
        f"{base_url}/api/v1/pipeline/load",
        json={"pipeline_ids": [pipeline_id]},
        timeout=10.0,
    )
    r.raise_for_status()
    # Wait for load to complete.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        s = requests.get(f"{base_url}/api/v1/pipeline/status", timeout=5.0).json()
        if s.get("status") == "loaded":
            break
        time.sleep(0.2)
    start = requests.post(
        f"{base_url}/api/v1/session/start",
        json={
            "pipeline_id": pipeline_id,
            "input_mode": "video",
            "input_source": {
                "enabled": False,
                "source_type": "video_file",
                "source_name": "",
            },
        },
        timeout=10.0,
    )
    start.raise_for_status()


@pytest.mark.chaos
@pytest.mark.graph
def test_workflow_switching_local(
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Onboard → for each of 4 swaps, load a different pipeline and verify frames."""
    report.metadata["workflow"] = "local-passthrough"

    flows.complete_onboarding_local(driver, workflow_id="local-passthrough")
    flows.start_stream_and_wait_first_frame(driver, timeout_ms=60_000)

    swap_count = {"done": 0, "failed": 0}
    for i in range(4):
        target = CPU_PIPELINES[i % len(CPU_PIPELINES)]
        try:
            failure_watcher.mark_initiated_stop()
            _swap_pipeline(scope_harness.base_url, target)
            swap_count["done"] += 1
        except Exception as e:
            swap_count["failed"] += 1
            report.fail(f"swap #{i} to {target} failed: {e}")

    report.measure("workflow_swaps_ok", swap_count["done"])
    report.measure("workflow_swaps_failed", swap_count["failed"])

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    # Best-effort cleanup.
    try:
        requests.post(f"{scope_harness.base_url}/api/v1/session/stop", timeout=5.0)
    except Exception:
        pass

    assert report.passed, f"Hard fails: {report.hard_fails}"
