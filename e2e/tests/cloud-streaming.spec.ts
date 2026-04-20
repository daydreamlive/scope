import { test, expect, Page } from "@playwright/test";

/**
 * E2E tests for Scope cloud streaming via fal.ai.
 *
 * The app is started with:
 *   VITE_DAYDREAM_API_KEY=... → baked into the frontend, makes the app
 *                              behave as signed-in so the cloud toggle
 *                              is enabled
 *   SCOPE_CLOUD_APP_ID=daydream/<app>/ws → points scope at a fal deploy
 *
 * Flow:
 * 1. App loads (already logged in via baked-in API key)
 * 2. Switch to Perform mode (default is Workflow/graph mode after the
 *    graph-mode redesign)
 * 3. Toggle Remote Inference on from the settings dialog
 * 4. Wait for cloud connection (Connection ID rendered)
 * 5. Select the passthrough pipeline
 * 6. Click the play overlay to start the stream
 * 7. Verify the output <video> is actually playing
 * 8. Stop the stream
 */

test.describe("Cloud Streaming", () => {
  test("connects to cloud and runs passthrough stream", async ({ page }) => {
    // Increase timeout for this test — cold-start on fal can take ~2min
    test.setTimeout(240000);

    // Mock the onboarding status API to skip onboarding.
    await page.route("**/api/v1/onboarding/status", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ completed: true, inference_mode: null }),
        });
      } else {
        await route.fulfill({ status: 200, body: "{}" });
      }
    });

    await page.goto("/");
    await page.waitForLoadState("domcontentloaded");

    // App is loaded once the Workflow/Perform mode toggle is present.
    const performToggle = page.locator('[aria-label="Perform Mode"]');
    await expect(performToggle).toBeVisible({ timeout: 15000 });
    await page.screenshot({ path: "test-results/01-initial-load.png" });

    // Step 1: Switch to Perform mode. Default after the graph-mode
    // redesign is Workflow; Perform is where the cloud toggle,
    // pipeline selector, and start button live.
    await performToggle.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: "test-results/02-perform-mode.png" });

    // Step 2: Enable cloud mode via settings dialog
    await enableCloudMode(page);

    // Step 3: Wait for cloud connection (cold-start can be slow)
    await waitForCloudConnection(page);

    // Step 4: Select passthrough pipeline
    await selectPassthroughModel(page);

    // Step 5: Switch input source to Camera so getUserMedia() fires.
    // Combined with the --use-fake-device-for-media-stream launch flag
    // (see playwright.config.ts), this gives the browser a real
    // MediaStreamTrack, which lets the browser↔local-scope WebRTC
    // actually deliver frames — which is what triggers CloudTrack
    // to call start_webrtc() and send the start_stream trickle
    // message to the runner.
    await selectCameraInput(page);

    // Step 6: Start streaming
    await startStream(page);

    // Step 7: Verify the OUTPUT video is actually playing (frames
    // round-tripped through the livepeer runner). Checking only
    // "any video is playing" would false-positive on the input.
    await verifyOutputStreamProcessing(page);

    // Step 8: Stop stream
    await stopStream(page);

    console.log("✅ Cloud streaming test passed");
  });
});

/**
 * Open settings via the cloud button in the header and toggle the
 * Remote Inference switch on.
 */
async function enableCloudMode(page: Page) {
  console.log("Enabling cloud mode...");

  // The cloud button in the header has title "Connect to cloud" (or
  // "Cloud connected" once active). Match by title so we find it in
  // any state.
  const cloudButton = page.locator(
    'button[title="Connect to cloud"], button[title="Cloud connected"], button[title="Connecting to cloud..."]'
  );
  await expect(cloudButton).toBeVisible({ timeout: 10000 });
  await cloudButton.click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: "test-results/03-settings-opened.png" });

  // The Remote Inference switch lives inside the settings dialog's
  // account tab.
  const cloudToggle = page.locator('[data-testid="cloud-toggle"]');
  await expect(cloudToggle).toBeVisible({ timeout: 10000 });
  await expect(cloudToggle).toBeEnabled({ timeout: 30000 });

  const checked = await cloudToggle.getAttribute("aria-checked");
  if (checked !== "true") {
    await cloudToggle.click();
    await expect(cloudToggle).toHaveAttribute("aria-checked", "true", {
      timeout: 10000,
    });
  }

  await page.screenshot({ path: "test-results/04-cloud-toggled.png" });
  console.log("✅ Cloud mode toggled on");
}

/**
 * Connection ID text only renders once `status.connected` is true.
 * Cold starts on fal can take ~2 minutes.
 */
async function waitForCloudConnection(page: Page) {
  console.log("Waiting for cloud connection...");

  await expect(page.getByText(/connection id/i)).toBeVisible({
    timeout: 180000,
  });
  await page.screenshot({ path: "test-results/05-cloud-connected.png" });
  console.log("✅ Cloud connection established");

  // Close the settings dialog so the Perform UI is fully interactive.
  await page.keyboard.press("Escape");
  await page.waitForTimeout(500);
}

/**
 * Select the passthrough pipeline from the Pipeline ID selector in
 * the Settings panel (Perform mode).
 */
async function selectPassthroughModel(page: Page) {
  console.log("Selecting passthrough model...");

  // "Pipeline ID" is an <h3>; its Radix <Select> trigger is the
  // combobox in the same surrounding container.
  const pipelineSection = page
    .locator("h3")
    .filter({ hasText: /^Pipeline ID$/ })
    .locator("..");
  const selectTrigger = pipelineSection.getByRole("combobox");

  await expect(selectTrigger).toBeVisible({ timeout: 10000 });
  await selectTrigger.click();

  const passthroughOption = page.getByRole("option", {
    name: /passthrough/i,
  });
  await expect(passthroughOption).toBeVisible({ timeout: 5000 });
  await passthroughOption.click();

  // Wait a moment for the pipeline to swap in the UI (loading state,
  // config form refresh).
  await page.waitForTimeout(1500);
  await page.screenshot({ path: "test-results/06-model-selected.png" });
  console.log("✅ Passthrough model selected");
}

/**
 * Start button is a PlayOverlay rendered with
 * data-testid="start-stream-button". Retry a few times — the overlay
 * can intercept clicks while the input video is still loading.
 */
async function startStream(page: Page) {
  console.log("Starting stream...");

  const startButton = page.locator('[data-testid="start-stream-button"]');

  const MAX_ATTEMPTS = 5;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    await expect(startButton).toBeVisible({ timeout: 10000 });
    await startButton.click();
    await page.waitForTimeout(2000);

    const stillVisible = await startButton.isVisible().catch(() => false);
    if (!stillVisible) {
      break;
    }

    console.log(
      `⚠️ Start button still visible after click (attempt ${attempt}/${MAX_ATTEMPTS}), retrying...`
    );
    await page.screenshot({
      path: `test-results/07-stream-retry-${attempt}.png`,
    });

    if (attempt === MAX_ATTEMPTS) {
      throw new Error(
        "Start stream button still visible after max retries — input video may not have loaded"
      );
    }
    await page.waitForTimeout(3000);
  }

  await page.waitForTimeout(2000);
  await page.screenshot({ path: "test-results/07-stream-started.png" });
  console.log("✅ Stream started");
}

/**
 * Switch the input source to Camera. Combined with the
 * --use-fake-device-for-media-stream browser flag, this gives the
 * browser a synthetic MediaStreamTrack via getUserMedia(), which is
 * what enables a real WebRTC peer connection between the browser and
 * local scope — the trigger for CloudTrack.start_webrtc() and the
 * runner's start_stream control message in Livepeer mode.
 */
async function selectCameraInput(page: Page) {
  console.log("Switching input source to Camera...");
  const cameraToggle = page.locator('[aria-label="Camera"]');
  await expect(cameraToggle).toBeVisible({ timeout: 10000 });
  await cameraToggle.click();
  // Give the app a moment to request getUserMedia and attach the
  // resulting stream to the input video element.
  await page.waitForTimeout(2000);
  await page.screenshot({ path: "test-results/06b-camera-selected.png" });
  console.log("✅ Camera input selected");
}

/**
 * Verify the *output* video inside the "Video Output" card is actually
 * playing — i.e., frames round-tripped through the livepeer runner and
 * came back to the browser. Checking any <video> would false-positive
 * on the local input preview.
 */
async function verifyOutputStreamProcessing(page: Page) {
  console.log("Verifying output stream processing...");

  // The Video Output card owns the output <video>. The element is
  // only rendered when `remoteStream` is set, so waiting for it to be
  // visible implicitly waits for the stream to come up.
  const outputCard = page
    .locator("text=Video Output")
    .locator("..")
    .locator("..");
  const outputVideo = outputCard.locator("video");

  await expect(outputVideo).toBeVisible({ timeout: 120000 });
  await page.screenshot({ path: "test-results/08a-output-rendered.png" });

  // Poll until the output video is actually playing with a non-zero
  // currentTime (frames arriving, not just the element attached).
  const MAX_WAIT_MS = 60000;
  const POLL_MS = 2000;
  const start = Date.now();

  while (Date.now() - start < MAX_WAIT_MS) {
    const playing = await outputVideo.evaluate((el) => {
      const v = el as HTMLVideoElement;
      return !v.paused && v.readyState >= 2 && v.currentTime > 0;
    });
    if (playing) {
      await page.screenshot({ path: "test-results/08b-frames-flowing.png" });
      console.log("✅ Output frames flowing");
      // Let the stream run briefly so stream_heartbeat events fire
      // on the runner side (frame_processor.py:707 emits roughly
      // every ~10s while the FrameProcessor is running).
      await page.waitForTimeout(15000);
      return;
    }
    await page.waitForTimeout(POLL_MS);
  }

  await page.screenshot({ path: "test-results/08c-no-output-frames.png" });
  throw new Error(
    `Output <video> element present but not playing after ${MAX_WAIT_MS}ms — frames not round-tripping`
  );
}

/**
 * Click the start-stream-button again to stop (it's a toggle — the
 * PlayOverlay turns into a stop overlay when the stream is running),
 * with a fallback to a button with a stop-like aria-label.
 */
async function stopStream(page: Page) {
  console.log("Stopping stream...");

  const stopOverlay = page.locator('[data-testid="start-stream-button"]');
  if (await stopOverlay.isVisible().catch(() => false)) {
    await stopOverlay.click();
  } else {
    const stopButton = page.getByRole("button", { name: /stop/i });
    if (await stopButton.isVisible().catch(() => false)) {
      await stopButton.click();
    }
  }
  await page.waitForTimeout(1000);
  await page.screenshot({ path: "test-results/09-stream-stopped.png" });
  console.log("✅ Stream stopped");
}
