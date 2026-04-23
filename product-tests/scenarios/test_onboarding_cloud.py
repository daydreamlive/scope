"""Onboarding smoke — cloud inference via a PR-deployed fal app.

Skipped unless ``SCOPE_CLOUD_APP_ID`` is set. PR CI sets this from the
``deploy-PR-to-fal`` workflow output; nightly pins it to a latest-main
fal app.

The cloud_auth phase is bypassed by the ``driver`` fixture, which seeds
a test auth blob into localStorage when @pytest.mark.cloud is present
(see ``harness/cloud_auth.py``). The app's ``isAuthenticated()`` check
reads that blob and auto-advances past sign-in.

The workflow matrix here mirrors the onboarding starters users actually
pick: mythical-creature (LongLive), ref-image (LTX image), ltx-text-to-video.
On the PR gate we run ``starter-mythical-creature`` only. Nightly runs all.
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


# Which cloud workflows run depends on the CI ring:
#   - PR gate sets SCOPE_CLOUD_RING=pr → cheapest only
#   - Nightly sets SCOPE_CLOUD_RING=nightly → full set
def _cloud_workflows() -> list[str]:
    ring = os.environ.get("SCOPE_CLOUD_RING", "pr")
    if ring == "nightly":
        return [
            "starter-mythical-creature",
            "starter-ref-image",
            "starter-ltx-text-to-video",
        ]
    return ["starter-mythical-creature"]  # one workflow only on the PR gate


@pytest.mark.cloud
@pytest.mark.parametrize("workflow_id", _cloud_workflows())
def test_onboarding_cloud(
    workflow_id: str,
    scope_harness: ScopeHarness,
    driver: PlaywrightDriver,
    retry_probe: RetryProbe,
    failure_watcher: FailureWatcher,
    report: TestReport,
):
    """Cold-start → pick cloud → pick workflow → Run → first frame."""
    report.metadata["workflow"] = workflow_id

    # The driver fixture auto-installs the cloud auth bypass init script,
    # so localStorage has a valid-shaped daydream_auth blob before app load.
    # Tests that exercise the real sign-in flow should not use this marker.

    flows.complete_onboarding_cloud(driver, workflow_id=workflow_id)

    first_ms = flows.start_stream_and_wait_first_frame(driver, timeout_ms=120_000)
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
