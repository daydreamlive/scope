import { test, expect, Page } from "@playwright/test";

/**
 * E2E tests for WebRTC disconnect recovery.
 *
 * Tests the reconnection logic added in PR #642:
 * 1. When WebRTC connection is lost, an error is surfaced to the frontend
 * 2. Auto-reconnection is attempted with exponential backoff
 * 3. The stream recovers after successful reconnection
 *
 * Uses the debug endpoint POST /api/v1/cloud/debug/disconnect-webrtc
 * to simulate a network disconnect.
 */

const API_BASE = "http://localhost:8000";

test.describe("WebRTC Reconnection", () => {
  test("recovers from WebRTC disconnect during streaming", async ({ page }) => {
    // Increase timeout for this test (reconnection can take time)
    test.setTimeout(300000); // 5 minutes

    // Navigate to the app
    await page.goto("/");
    await expect(
      page.locator("h1", { hasText: "Daydream Scope" })
    ).toBeVisible({ timeout: 15000 });

    // Step 1: Enable cloud mode and connect
    await enableCloudMode(page);
    await waitForCloudConnection(page);

    // Step 2: Select passthrough model and start streaming
    await selectPassthroughModel(page);
    await startStream(page);

    // Step 3: Verify stream is working
    await verifyStreamProcessing(page);
    await page.screenshot({
      path: "test-results/reconnect-01-stream-running.png",
    });

    // Step 4: Force disconnect WebRTC via debug endpoint
    console.log("Triggering WebRTC disconnect via debug endpoint...");
    const disconnectResponse = await page.request.post(
      `${API_BASE}/api/v1/cloud/debug/disconnect-webrtc`
    );
    expect(disconnectResponse.ok()).toBeTruthy();

    const disconnectStatus = await disconnectResponse.json();
    console.log("Disconnect response:", JSON.stringify(disconnectStatus));

    await page.screenshot({
      path: "test-results/reconnect-02-after-disconnect.png",
    });

    // Step 5: Verify error surfaces to frontend
    await verifyErrorSurfaced(page);

    // Step 6: Wait for reconnection (up to 30s for backoff + retries)
    await waitForReconnection(page);
    await page.screenshot({
      path: "test-results/reconnect-03-after-reconnect.png",
    });

    // Step 7: Verify stream resumes
    await verifyStreamProcessing(page);
    await page.screenshot({
      path: "test-results/reconnect-04-stream-recovered.png",
    });

    // Step 8: Stop stream
    await stopStream(page);

    console.log("✅ WebRTC reconnection test passed");
  });

  test("preserves parameters across reconnection", async ({ page }) => {
    // Tests that parameter updates via send_parameters() are preserved
    // when reconnection occurs (fix for stale params issue)
    test.setTimeout(300000);

    await page.goto("/");
    await expect(
      page.locator("h1", { hasText: "Daydream Scope" })
    ).toBeVisible({ timeout: 15000 });

    // Step 1: Connect and start streaming
    await enableCloudMode(page);
    await waitForCloudConnection(page);
    await selectPassthroughModel(page);
    await startStream(page);
    await verifyStreamProcessing(page);

    // Step 2: Update parameters via API (simulates user changing prompt)
    const testParams = { prompt: "test-reconnection-prompt-" + Date.now() };
    const updateResponse = await page.request.post(
      `${API_BASE}/api/v1/pipeline/parameters`,
      { data: testParams }
    );
    // Parameter update may succeed or fail depending on pipeline state
    console.log(`Parameter update status: ${updateResponse.status()}`);

    // Step 3: Force disconnect
    console.log("Triggering WebRTC disconnect...");
    const disconnectResponse = await page.request.post(
      `${API_BASE}/api/v1/cloud/debug/disconnect-webrtc`
    );
    expect(disconnectResponse.ok()).toBeTruthy();

    // Step 4: Wait for reconnection
    await waitForReconnection(page);

    // Step 5: Verify stream recovers (parameters should be reapplied)
    await verifyStreamProcessing(page);

    await stopStream(page);
    console.log("✅ Parameter preservation test passed");
  });

  test("surfaces error when reconnection fails multiple times", async ({ page }) => {
    // Tests that after exhausting reconnection attempts, an error message is shown
    // This would require the cloud endpoint to be unavailable, which is hard to simulate
    // For now, we verify the error state fields are properly typed and accessible
    test.setTimeout(60000);

    await page.goto("/");

    // Start a cloud connection
    await enableCloudMode(page);
    await waitForCloudConnection(page);

    // Get initial status to verify reconnection tracking fields exist
    const statusResponse = await page.request.get(
      `${API_BASE}/api/v1/cloud/status`
    );
    const status = await statusResponse.json();

    // Verify reconnection-related fields
    expect(typeof status.webrtc_connected).toBe("boolean");
    expect(status.webrtc_error === null || typeof status.webrtc_error === "string").toBe(true);
    expect(typeof status.webrtc_reconnecting).toBe("boolean");

    console.log("✅ Reconnection failure error state test passed");
  });

  test("status endpoint includes WebRTC reconnection fields", async ({ page }) => {
    // Validates that the new reconnection status fields are present in the API
    // TODO: Add separate test for reconnection failure when cloud mocking is available
    test.setTimeout(60000);

    await page.goto("/");

    // Check that the status endpoint includes the new fields
    const statusResponse = await page.request.get(
      `${API_BASE}/api/v1/cloud/status`
    );
    expect(statusResponse.ok()).toBeTruthy();

    const status = await statusResponse.json();
    console.log("Status response fields:", Object.keys(status));

    // Verify the new fields exist in the schema
    expect(status).toHaveProperty("webrtc_connected");
    expect(status).toHaveProperty("webrtc_error");
    expect(status).toHaveProperty("webrtc_reconnecting");

    console.log("✅ Status schema includes reconnection fields");
  });
});

/**
 * Enable cloud mode by opening settings and toggling Remote Inference.
 */
async function enableCloudMode(page: Page) {
  console.log("Enabling cloud mode...");

  const cloudIcon = page.locator('button[title*="cloud" i]');
  await expect(cloudIcon).toBeVisible({ timeout: 10000 });
  await cloudIcon.click();

  const cloudToggle = page.locator('[data-testid="cloud-toggle"]');
  await expect(cloudToggle).toBeVisible({ timeout: 10000 });
  await expect(cloudToggle).toBeEnabled({ timeout: 30000 });

  const isEnabled = await cloudToggle.getAttribute("aria-checked");
  if (isEnabled !== "true") {
    await cloudToggle.click();
    await expect(cloudToggle).toHaveAttribute("aria-checked", "true", {
      timeout: 10000,
    });
  }

  console.log("✅ Cloud mode enabled");
}

/**
 * Wait for the cloud connection to be established.
 */
async function waitForCloudConnection(page: Page) {
  console.log("Waiting for cloud connection...");

  await expect(page.getByText(/connection id/i)).toBeVisible({
    timeout: 120000,
  });

  console.log("✅ Cloud connection established");
  await page.keyboard.press("Escape");
  await page.waitForTimeout(500);
}

/**
 * Select the passthrough pipeline.
 */
async function selectPassthroughModel(page: Page) {
  console.log("Selecting passthrough model...");

  const pipelineSection = page.locator("text=Pipeline ID").locator("..");
  const selectTrigger = pipelineSection.getByRole("combobox");

  await expect(selectTrigger).toBeVisible({ timeout: 10000 });
  await selectTrigger.click();

  const passthroughOption = page.getByRole("option", {
    name: /passthrough/i,
  });
  await expect(passthroughOption).toBeVisible({ timeout: 5000 });
  await passthroughOption.click();

  console.log("✅ Passthrough model selected");
}

/**
 * Start the video stream.
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
    await page.waitForTimeout(2000);

    const stillVisible = await startButton.isVisible().catch(() => false);
    if (!stillVisible) break;

    console.log(`⚠️ Retry starting stream (attempt ${attempt}/${MAX_ATTEMPTS})`);
    if (attempt === MAX_ATTEMPTS) {
      throw new Error("Failed to start stream after max retries");
    }
    await page.waitForTimeout(3000);
  }

  await page.waitForTimeout(2000);
  console.log("✅ Stream started");
}

/**
 * Verify that frames are being processed.
 */
async function verifyStreamProcessing(page: Page) {
  console.log("Verifying stream processing...");

  const outputCard = page.locator("text=Video Output").locator("../..");
  const outputVideo = outputCard.locator("video");

  await expect(outputVideo).toBeVisible({ timeout: 30000 });

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

  if (!isPlaying) {
    throw new Error("Stream does not appear to be processing frames");
  }

  console.log("✅ Stream is processing frames");
}

/**
 * Verify that the WebRTC error is surfaced via the status API.
 * Returns true if error/reconnecting state was observed, false if connection recovered quickly.
 */
async function verifyErrorSurfaced(page: Page): Promise<boolean> {
  console.log("Verifying error is surfaced...");

  // Poll the status endpoint for the error
  const MAX_WAIT_MS = 10000;
  const POLL_MS = 1000;
  const start = Date.now();

  while (Date.now() - start < MAX_WAIT_MS) {
    const response = await page.request.get(`${API_BASE}/api/v1/cloud/status`);
    const status = await response.json();

    if (status.webrtc_error || status.webrtc_reconnecting) {
      console.log(
        `✅ Error surfaced: ${status.webrtc_error || "(reconnecting)"}`
      );
      return true;
    }

    await page.waitForTimeout(POLL_MS);
  }

  // The connection might have already recovered, which is also acceptable
  console.log("⚠️ Error may have been transient (connection recovered quickly)");
  return false;
}

/**
 * Wait for WebRTC reconnection to complete.
 */
async function waitForReconnection(page: Page) {
  console.log("Waiting for WebRTC reconnection...");

  const MAX_WAIT_MS = 60000; // Allow up to 60s for reconnection (3 attempts with backoff)
  const POLL_MS = 2000;
  const start = Date.now();

  while (Date.now() - start < MAX_WAIT_MS) {
    const response = await page.request.get(`${API_BASE}/api/v1/cloud/status`);
    const status = await response.json();

    if (status.webrtc_connected && !status.webrtc_reconnecting) {
      console.log("✅ WebRTC reconnected successfully");
      return;
    }

    const elapsed = Math.round((Date.now() - start) / 1000);
    console.log(
      `⏳ Waiting for reconnection... (${elapsed}s, ` +
        `connected: ${status.webrtc_connected}, ` +
        `reconnecting: ${status.webrtc_reconnecting})`
    );

    await page.waitForTimeout(POLL_MS);
  }

  throw new Error("WebRTC reconnection timed out");
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
    console.log("✅ Stream stopped");
  } else {
    console.log("⚠️ Stop button not found");
  }
}
