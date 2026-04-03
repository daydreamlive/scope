import { test, expect, Page } from "@playwright/test";

/**
 * E2E tests for Scope cloud streaming via fal.ai.
 *
 * The app is started with:
 *   VITE_DAYDREAM_API_KEY=... → handles auth (shows as logged in)
 *   SCOPE_CLOUD_APP_ID=scope-pr-<N> → configures cloud endpoint
 *
 * These tests verify the full flow:
 * 1. App loads (already logged in via API key)
 * 2. Enable cloud mode
 * 3. Start a stream with the passthrough model
 * 4. Verify frames are being processed
 */

test.describe("Cloud Streaming", () => {
  test("connects to cloud and runs passthrough stream", async ({ page }) => {
    // Increase timeout for this test
    test.setTimeout(180000); // 3 minutes

    // Mock the onboarding status API to skip onboarding in e2e tests
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

    // Navigate to the app (running at localhost:8000)
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Daydream Scope", exact: true })
    ).toBeVisible({ timeout: 15000 });

    // Take screenshot after initial load — app loads directly into the streaming interface
    await page.screenshot({ path: "test-results/01-initial-load.png" });

    // Step 1: Enable cloud mode (endpoint is pre-configured via SCOPE_CLOUD_APP_ID)
    await enableCloudMode(page);

    // Step 2: Wait for cloud connection
    await waitForCloudConnection(page);

    // Step 3: Ensure app is in Perform Mode (Pipeline ID selector is only visible there)
    await ensurePerformMode(page);

    // Step 4: Select passthrough model
    await selectPassthroughModel(page);

    // Step 5: Start streaming
    await startStream(page);

    // Step 6: Verify frames are being processed
    await verifyStreamProcessing(page);

    // Step 7: Stop stream
    await stopStream(page);

    console.log("✅ Cloud streaming test passed");
  });
});

/**
 * Enable cloud mode by opening settings and toggling Remote Inference.
 */
async function enableCloudMode(page: Page) {
  console.log("Enabling cloud mode...");

  // Open settings dialog via the cloud icon in the header
  const cloudIcon = page.locator('button[title*="cloud" i], button[title*="remote inference" i]');
  await expect(cloudIcon).toBeVisible({ timeout: 10000 });
  await cloudIcon.click();

  await page.screenshot({ path: "test-results/03-settings-opened.png" });

  // Find the Remote Inference switch inside the settings dialog
  const cloudToggle = page.locator('[data-testid="cloud-toggle"]');
  await expect(cloudToggle).toBeVisible({ timeout: 10000 });

  // Wait for the toggle to be enabled (auth may still be initializing)
  await expect(cloudToggle).toBeEnabled({ timeout: 30000 });

  // Toggle on if not already enabled
  const isEnabled = await cloudToggle.getAttribute("aria-checked");
  if (isEnabled !== "true") {
    await cloudToggle.click();
    await expect(cloudToggle).toHaveAttribute("aria-checked", "true", {
      timeout: 10000,
    });
  }

  await page.screenshot({ path: "test-results/04-cloud-toggled.png" });
  console.log("✅ Cloud mode enabled");
}

/**
 * Wait for the cloud connection to be established.
 * The Connection ID element only appears when status.connected is true.
 */
async function waitForCloudConnection(page: Page) {
  console.log("Waiting for cloud connection...");

  // The Connection ID text only renders when connected, so wait for it
  await expect(page.getByText(/connection id/i)).toBeVisible({
    timeout: 120000,
  });

  await page.screenshot({ path: "test-results/05-cloud-connected.png" });
  console.log("✅ Cloud connection established");

  // Close the settings dialog
  await page.keyboard.press("Escape");
  await page.waitForTimeout(500);
}

/**
 * Ensure the app is in Perform Mode (not Graph/Workflow Builder mode).
 * The app defaults to Graph Mode; the SettingsPanel with Pipeline ID is only
 * visible in Perform Mode. If the "Perform Mode" button is present in the
 * header, click it to switch.
 */
async function ensurePerformMode(page: Page) {
  console.log("Ensuring Perform Mode...");

  // The header button reads "Perform Mode" when the app is in Graph Mode
  // (clicking it switches to Perform Mode). If it reads "Workflow Builder",
  // we're already in Perform Mode.
  const performModeButton = page.getByRole("button", { name: /perform mode/i });

  if (await performModeButton.isVisible({ timeout: 3000 }).catch(() => false)) {
    await performModeButton.click();
    // Wait for the SettingsPanel to appear (Pipeline ID heading)
    await expect(page.locator("text=Pipeline ID")).toBeVisible({
      timeout: 10000,
    });
    await page.screenshot({ path: "test-results/05b-perform-mode.png" });
    console.log("✅ Switched to Perform Mode");
  } else {
    console.log("✅ Already in Perform Mode");
  }
}

/**
 * Select the passthrough pipeline in the Settings panel.
 * The pipeline selector is a Radix Select with heading "Pipeline ID".
 */
async function selectPassthroughModel(page: Page) {
  console.log("Selecting passthrough model...");

  // The pipeline selector trigger shows the current pipeline name.
  // Find it via the "Pipeline ID" heading's sibling combobox.
  const pipelineSection = page.locator("text=Pipeline ID").locator("..");
  const selectTrigger = pipelineSection.getByRole("combobox");

  await expect(selectTrigger).toBeVisible({ timeout: 10000 });
  await selectTrigger.click();

  // Wait for dropdown and select passthrough
  const passthroughOption = page.getByRole("option", {
    name: /passthrough/i,
  });
  await expect(passthroughOption).toBeVisible({ timeout: 5000 });
  await passthroughOption.click();

  await page.screenshot({ path: "test-results/06-model-selected.png" });
  console.log("✅ Passthrough model selected");
}

/**
 * Start the video stream, retrying if the input video hasn't loaded yet.
 * If the play button is still visible after clicking (i.e. we didn't
 * transition to a connecting/loading state), try again.
 */
async function startStream(page: Page) {
  console.log("Starting stream...");

  const startButton = page
    .locator('[data-testid="start-stream-button"]')
    .or(page.getByRole("button", { name: /start stream/i }));

  const MAX_ATTEMPTS = 5;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    await expect(startButton).toBeVisible({ timeout: 10000 });
    await startButton.click();

    // Give the app a moment to react
    await page.waitForTimeout(2000);

    // If the play button disappeared, the stream is starting
    const stillVisible = await startButton.isVisible().catch(() => false);
    if (!stillVisible) {
      break;
    }

    console.log(
      `⚠️ Play button still visible after click (attempt ${attempt}/${MAX_ATTEMPTS}), retrying...`
    );
    await page.screenshot({
      path: `test-results/07-stream-retry-${attempt}.png`,
    });

    if (attempt === MAX_ATTEMPTS) {
      throw new Error(
        "Start stream button still visible after max retries — input video may not have loaded"
      );
    }

    // Wait before retrying
    await page.waitForTimeout(3000);
  }

  await page.waitForTimeout(2000);
  await page.screenshot({ path: "test-results/07-stream-started.png" });
  console.log("✅ Stream started");
}

/**
 * Verify that frames are being processed by the cloud.
 * The output video is inside the "Video Output" card and only renders
 * when a remoteStream is active.
 */
async function verifyStreamProcessing(page: Page) {
  console.log("Verifying stream processing...");

  // The output <video> element is inside the "Video Output" card
  const outputCard = page.locator("text=Video Output").locator("../..");
  const outputVideo = outputCard.locator("video");

  // Wait for the video element to appear (stream needs to produce frames)
  await expect(outputVideo).toBeVisible({ timeout: 30000 });

  // Poll until the video is actually playing (may take a few seconds for
  // WebRTC negotiation and first frame to arrive)
  const MAX_WAIT_MS = 30000;
  const POLL_MS = 2000;
  const start = Date.now();
  let isPlaying = false;

  while (Date.now() - start < MAX_WAIT_MS) {
    isPlaying = await outputVideo.evaluate((el) => {
      const v = el as HTMLVideoElement;
      return !v.paused && v.readyState >= 2;
    });
    if (isPlaying) break;
    await page.waitForTimeout(POLL_MS);
  }

  await page.screenshot({ path: "test-results/08-stream-running.png" });

  if (!isPlaying) {
    throw new Error("Stream does not appear to be processing frames");
  }

  console.log("✅ Stream is processing frames");
}

/**
 * Stop the stream.
 */
async function stopStream(page: Page) {
  console.log("Stopping stream...");

  const stopButton = page
    .getByRole("button", { name: /stop|end|pause/i })
    .or(page.locator('[data-testid="stop-stream-button"]'));

  if (await stopButton.isVisible({ timeout: 5000 }).catch(() => false)) {
    await stopButton.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: "test-results/09-stream-stopped.png" });
    console.log("✅ Stream stopped");
  } else {
    console.log("⚠️ Stop button not found, stream may auto-stop");
  }
}
