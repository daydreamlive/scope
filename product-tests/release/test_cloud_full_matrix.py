"""Nightly release-gate — cloud full-matrix, all three starter workflows.

Mirrors the intent of the retired ``e2e/`` TypeScript scaffold: a
cloud-connected run of every starter workflow users actually pick in
onboarding, with first-frame SLOs enforced.

Only runs in the nightly ring (``SCOPE_CLOUD_RING=nightly``). On the PR
gate, ``test_onboarding_cloud.py[starter-mythical-creature]`` is enough
signal.
"""

from __future__ import annotations

import os

import pytest
from harness import baselines, flows, gates
from harness.driver import PlaywrightDriver
from harness.failure_watcher import FailureWatcher
from harness.report import TestReport
from harness.retry_probe import RetryProbe
from harness.scope_process import ScopeHarness

STARTER_WORKFLOWS = [
    "starter-mythical-creature",
    "starter-ref-image",
    "starter-ltx-text-to-video",
]


@pytest.mark.onboarding
@pytest.mark.cloud
@pytest.mark.slow
@pytest.mark.parametrize("workflow_id", STARTER_WORKFLOWS)
def test_cloud_full_matrix(
    workflow_id: str,
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    if os.environ.get("SCOPE_CLOUD_RING", "pr") != "nightly":
        pytest.skip("release-gate full-matrix runs only in nightly ring")

    report.metadata["workflow"] = workflow_id

    flows.complete_onboarding_cloud(driver, workflow_id=workflow_id)
    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=180_000)
    baselines.check(
        report,
        "cloud",
        workflow_id.removeprefix("starter-"),
        "first_frame_time_ms",
        int(first_ms),
    )

    gates.enforce_all_gates(report, retry_probe, failure_watcher, driver)

    failure_watcher.mark_initiated_stop()
    flows.stop_stream(driver)

    assert report.passed, f"Hard fails: {report.hard_fails}"
