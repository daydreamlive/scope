"""Cloud streaming smoke — Perform mode + camera input + output frames flowing.

Ported from ``e2e/tests/cloud-streaming.spec.ts`` (originally PR #962, Emran).
This is the canonical end-to-end "did the deployed fal app actually work"
check. Distinct from ``test_onboarding_cloud.py`` — that one drives the
onboarding flow; this one drives Perform mode with a synthetic camera and
asserts that round-tripped frames render in the output video element.

Triggered by the ``testing-livepeer-fal-deploy`` skill (the "test cloud"
trigger) plus by CI nightly. Skips when ``SCOPE_CLOUD_APP_ID`` is unset.

Flow:
  1. Mock ``onboarding/status`` to skip onboarding.
  2. Switch to Perform mode (default after the graph-mode redesign).
  3. Toggle Remote Inference ON in the settings dialog.
  4. Wait for cloud connection (Connection ID rendered) — cold start ≤2min.
  5. Select the passthrough pipeline.
  6. Switch input source to Camera (synthetic via launch args).
  7. Click the start-stream-button.
  8. Verify the *output* ``<video>`` is actually playing
     (currentTime > 0, readyState >= 2).
  9. Stop stream.
"""

from __future__ import annotations

import time

from harness.scenario import scenario
from playwright.sync_api import Page, expect


def _mock_onboarding_status(page: Page) -> None:
    """Mock onboarding/status so the app skips straight to the main UI."""

    def _handler(route):
        if route.request.method == "GET":
            route.fulfill(
                status=200,
                content_type="application/json",
                body='{"completed": true, "inference_mode": null}',
            )
        else:
            route.fulfill(status=200, body="{}")

    page.route("**/api/v1/onboarding/status", _handler)


def _switch_to_perform_mode(page: Page) -> None:
    """Default after the graph-mode redesign is Workflow; Perform is where
    the cloud toggle, pipeline selector, and start button live."""
    perform_toggle = page.locator('[aria-label="Perform Mode"]')
    expect(perform_toggle).to_be_visible(timeout=15_000)
    perform_toggle.click()
    page.wait_for_timeout(1000)


def _enable_cloud_mode(page: Page) -> None:
    """Open settings via the cloud button in the header and toggle the
    Remote Inference switch on."""
    # Cloud button title varies by state — match any.
    cloud_button = page.locator(
        'button[title="Connect to cloud"], button[title="Cloud connected"], '
        'button[title="Connecting to cloud..."]'
    )
    expect(cloud_button).to_be_visible(timeout=10_000)
    cloud_button.click()
    page.wait_for_timeout(500)

    cloud_toggle = page.locator('[data-testid="cloud-toggle"]')
    expect(cloud_toggle).to_be_visible(timeout=10_000)
    expect(cloud_toggle).to_be_enabled(timeout=30_000)

    if cloud_toggle.get_attribute("aria-checked") != "true":
        cloud_toggle.click()
        expect(cloud_toggle).to_have_attribute("aria-checked", "true", timeout=10_000)


def _wait_for_cloud_connection(page: Page) -> None:
    """Connection ID text only renders once status.connected is true.
    Cold starts on fal can take ~2 minutes."""
    expect(page.get_by_text("connection id", exact=False)).to_be_visible(
        timeout=180_000
    )
    # Close the settings dialog so the Perform UI is fully interactive.
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)


def _select_passthrough(page: Page) -> None:
    """Select the passthrough pipeline from the Pipeline ID selector."""
    # "Pipeline ID" is an <h3>; its Radix <Select> trigger is the
    # combobox in the same surrounding container.
    pipeline_section = page.locator("h3").filter(has_text="Pipeline ID").locator("..")
    select_trigger = pipeline_section.get_by_role("combobox")
    expect(select_trigger).to_be_visible(timeout=10_000)
    select_trigger.click()

    passthrough_option = page.get_by_role("option", name="passthrough")
    expect(passthrough_option).to_be_visible(timeout=5_000)
    passthrough_option.click()

    # Let the pipeline swap settle in the UI.
    page.wait_for_timeout(1500)


def _select_camera_input(page: Page) -> None:
    """Switch the input source to Camera. Combined with the
    ``--use-fake-device-for-media-stream`` launch flag, this gives the
    browser a synthetic MediaStreamTrack via getUserMedia(), which is
    what enables a real WebRTC peer connection between the browser and
    local scope — the trigger for ``CloudTrack.start_webrtc()`` and the
    runner's ``start_stream`` control message in Livepeer mode."""
    camera_toggle = page.locator('[aria-label="Camera"]')
    expect(camera_toggle).to_be_visible(timeout=10_000)
    camera_toggle.click()
    # Brief settle so getUserMedia can attach the stream to the input video.
    page.wait_for_timeout(2000)


def _start_stream(page: Page) -> None:
    """Click the start-stream-button. Retry — the play overlay can
    intercept clicks while the input video is still loading."""
    start_button = page.locator('[data-testid="start-stream-button"]')
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        expect(start_button).to_be_visible(timeout=10_000)
        start_button.click()
        page.wait_for_timeout(2000)

        try:
            still_visible = start_button.is_visible()
        except Exception:
            still_visible = False
        if not still_visible:
            return

        if attempt == max_attempts:
            raise AssertionError(
                "start-stream-button still visible after max retries — "
                "input video may not have loaded"
            )
        page.wait_for_timeout(3000)


def _verify_output_playing(page: Page) -> None:
    """Verify the *output* video inside the 'Video Output' card is actually
    playing — i.e., frames round-tripped through the livepeer runner and
    came back to the browser. Checking any <video> would false-positive
    on the local input preview."""
    output_card = page.locator("text=Video Output").locator("..").locator("..")
    output_video = output_card.locator("video")
    expect(output_video).to_be_visible(timeout=120_000)

    # Poll until the output video has currentTime > 0 (frames arriving).
    max_wait_sec = 60
    poll_sec = 2
    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        playing = output_video.evaluate(
            """(el) => !el.paused && el.readyState >= 2 && el.currentTime > 0"""
        )
        if playing:
            # Let the stream run briefly so stream_heartbeat events fire
            # on the runner side.
            page.wait_for_timeout(15_000)
            return
        page.wait_for_timeout(poll_sec * 1000)

    raise AssertionError(
        f"output <video> present but not playing after {max_wait_sec}s — "
        "frames not round-tripping"
    )


def _stop_stream(page: Page) -> None:
    """Click the start-stream-button again to stop (it's a toggle), with
    a fallback to a button labeled stop."""
    stop_overlay = page.locator('[data-testid="start-stream-button"]')
    try:
        if stop_overlay.is_visible():
            stop_overlay.click()
            return
    except Exception:
        pass
    try:
        stop_button = page.get_by_role("button", name="stop")
        if stop_button.is_visible():
            stop_button.click()
    except Exception:
        pass


@scenario(
    mode="cloud",
    workflow="local-passthrough",
    feature=("ui", "lifecycle"),
)
def test_cloud_streaming_perform_mode_passthrough(ctx):
    """Cloud streaming end-to-end via Perform mode + synthetic camera.

    The canonical "did my fal deploy work?" check, runnable against any
    deployed fal app via ``SCOPE_CLOUD_APP_ID``. This is what the
    ``testing-livepeer-fal-deploy`` skill invokes when a user says
    "test cloud".
    """
    page = ctx.driver.page
    ctx.report.metadata["workflow"] = "perform-cloud-passthrough"

    _mock_onboarding_status(page)
    page.goto(ctx.base_url)
    page.wait_for_load_state("domcontentloaded")

    _switch_to_perform_mode(page)
    _enable_cloud_mode(page)
    _wait_for_cloud_connection(page)
    _select_passthrough(page)
    _select_camera_input(page)
    _start_stream(page)
    _verify_output_playing(page)

    # @scenario teardown auto-asserts: zero retries, zero unexpected
    # closes, zero UI errors. Stopping the stream cleanly here so the
    # teardown's mark_initiated_stop matches.
    ctx.failure_watcher.mark_initiated_stop()
    _stop_stream(page)
