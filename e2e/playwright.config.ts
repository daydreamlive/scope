import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for Scope E2E tests.
 *
 * The app is started locally with:
 *   VITE_DAYDREAM_API_KEY=... uv run build
 *   SCOPE_CLOUD_APP_ID=scope-pr-<N> uv run daydream-scope
 *
 * This runs the app at localhost:8000 with the API key handling auth
 * and SCOPE_CLOUD_APP_ID pointing to the fal deployment.
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: [
    ["html", { open: "never" }],
    ["list"],
    ...(process.env.CI ? [["github" as const]] : []),
  ],
  use: {
    baseURL: "http://localhost:8000",
    trace: "on-first-retry",
    screenshot: "on",
    video: "retain-on-failure",
    // Longer timeout for cloud operations
    actionTimeout: 30000,
    navigationTimeout: 60000,
    // Grant camera/mic so getUserMedia() succeeds without a UI prompt
    // (the browser launch flags below provide a synthetic feed).
    permissions: ["camera", "microphone"],
  },
  // Global timeout per test
  timeout: 300000, // 5 minutes (cold-start fal containers can run long)
  expect: {
    timeout: 30000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // Feed getUserMedia a synthetic video source so a real WebRTC
          // peer connection can complete end-to-end — without these
          // flags, headless Chromium has no camera and ICE stalls.
          args: [
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--auto-select-desktop-capture-source=fake",
          ],
        },
      },
    },
  ],
  // Output directory for test artifacts
  outputDir: "test-results/",
});
