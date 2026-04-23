"""Reusable high-level flows that compose PlaywrightDriver actions.

Scenarios should call these helpers instead of copy-pasting onboarding
click sequences. If the onboarding flow changes, update here once.
"""

from __future__ import annotations

from .driver import PlaywrightDriver

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
    driver: PlaywrightDriver, workflow_id: str = "local-passthrough"
) -> None:
    """Click through local-mode onboarding and dismiss the tour.

    Leaves the app at the graph view with the Run button visible.
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
