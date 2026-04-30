"""Reusable high-level flows that compose PlaywrightDriver actions.

Scenarios should call these helpers instead of copy-pasting onboarding
click sequences. If the onboarding flow changes, update here once.

Direct-HTTP test helpers (``http_*``) live here too so HTTP-only tests
have a one-stop shop. They don't take a driver and are usable from
tests that intentionally skip the UI.
"""

from __future__ import annotations

import time

import requests

from .driver import PlaywrightDriver

# ---------------------------------------------------------------------------
# Direct-HTTP helpers — for tests that don't go through the UI.
# ---------------------------------------------------------------------------
# UI-driven tests get pipeline-loading "for free" via onboarding. Direct-HTTP
# tests must do it themselves before ``session/start`` — otherwise the
# FrameProcessor fails to start with "Pipeline <id> not loaded".


def http_load_pipeline_and_wait(
    base_url: str,
    pipeline_ids: list[str],
    timeout_sec: float = 30.0,
) -> None:
    """Load pipelines via HTTP and poll until ``status == loaded``.

    Direct-HTTP tests (those that don't take a ``driver`` fixture) must
    call this before ``POST /api/v1/session/start``. The CLAUDE.md doc
    documents the same sequence (resolve → load → wait → start).

    Raises ``AssertionError`` on non-200 from ``pipeline/load`` or if the
    pipeline doesn't reach ``loaded`` within ``timeout_sec``.
    """
    r = requests.post(
        f"{base_url}/api/v1/pipeline/load",
        json={"pipeline_ids": list(pipeline_ids)},
        timeout=30.0,
    )
    assert r.status_code == 200, f"pipeline/load: {r.status_code} {r.text[:200]}"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        r = requests.get(f"{base_url}/api/v1/pipeline/status", timeout=10.0)
        r.raise_for_status()
        if r.json().get("status") == "loaded":
            return
        time.sleep(0.5)
    raise AssertionError(
        f"pipelines {pipeline_ids} did not reach loaded within {timeout_sec}s"
    )


# ---------------------------------------------------------------------------
# Workflow catalogue (subset used by product-tests)
# ---------------------------------------------------------------------------
# Source of truth: frontend/src/components/onboarding/starterWorkflows.ts

# Workflows reachable on CPU-only rings.
CPU_WORKFLOWS = {
    "local-passthrough",
}

# Workflows requiring GPU (nightly ring) or cloud relay.
GPU_OR_CLOUD_WORKFLOWS = {
    "starter-mythical-creature",
    "starter-ref-image",
    "starter-ltx-text-to-video",
}

ALL_WORKFLOWS = CPU_WORKFLOWS | GPU_OR_CLOUD_WORKFLOWS


def complete_onboarding_local(
    driver: PlaywrightDriver,
    workflow_id: str = "local-passthrough",
    dismiss_tour: bool = True,
) -> None:
    """Click through local-mode onboarding.

    Leaves the app at the graph view with the Run button visible. By default
    dismisses the tour popover (most tests want a clean graph). Pass
    ``dismiss_tour=False`` to leave the first tour step up — useful for tests
    that need to assert on the tour popover itself (e.g. positioning checks).
    """
    driver.click_testid("inference-mode-local")
    driver.click_testid("inference-mode-continue")

    # Telemetry disclosure — click Decline unless auto-advance beat us to it.
    try:
        driver.wait_testid("telemetry-decline", timeout_ms=3000)
        driver.click_testid("telemetry-decline")
    except Exception:
        pass

    driver.click_testid(f"workflow-card-{workflow_id}")
    driver.click_testid("workflow-get-started")

    # Workflow import dialog — confirm if it appears.
    try:
        driver.wait_testid("workflow-import-load", timeout_ms=5000)
        driver.click_testid("workflow-import-load")
    except Exception:
        pass

    if dismiss_tour:
        driver.click_all_tour_steps()
    driver.wait_testid("stream-run-stop")


def complete_onboarding_cloud(
    driver: PlaywrightDriver, workflow_id: str = "starter-mythical-creature"
) -> None:
    """Cloud-mode onboarding.

    Only usable when a test-only auth bypass is in effect (see
    ``cloud_auth_bypass`` fixture). The caller is responsible for ensuring
    the backend is configured to skip sign-in.
    """
    driver.click_testid("inference-mode-cloud")
    driver.click_testid("inference-mode-continue")

    # Cloud auth step is bypassed by fixture; wait for cloud_connecting
    # overlay to clear into the workflow picker.
    driver.wait_testid(f"workflow-card-{workflow_id}", timeout_ms=60_000)
    driver.click_testid(f"workflow-card-{workflow_id}")
    driver.click_testid("workflow-get-started")

    try:
        driver.wait_testid("workflow-import-load", timeout_ms=5000)
        driver.click_testid("workflow-import-load")
    except Exception:
        pass

    driver.click_all_tour_steps()
    driver.wait_testid("stream-run-stop")


def start_stream_and_wait_first_frame(
    driver: PlaywrightDriver, timeout_ms: int = 90_000
) -> float:
    """Click Run; return ms to first rendered video frame."""
    driver.click_testid("stream-run-stop")
    return driver.wait_first_frame(timeout_ms=timeout_ms)


def stop_stream(driver: PlaywrightDriver) -> None:
    """Click Stop if streaming is active. No-op otherwise."""
    try:
        loc = driver.page.locator(
            '[data-testid="stream-run-stop"][data-streaming="true"]'
        )
        if loc.count() > 0:
            driver.click_testid("stream-run-stop")
    except Exception:
        pass
