import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for Scope E2E tests.
 *
 * The app is started locally with:
 *   VITE_DAYDREAM_API_KEY=... uv run build
 *   SCOPE_CLOUD_APP_ID=daydream/scope-livepeer-pr-<N>--preview/ws uv run daydream-scope
 *
 * This runs the app at localhost:8000 with the API key handling auth
 * and SCOPE_CLOUD_APP_ID pointing to the Livepeer fal deployment.
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
  },
  // Global timeout per test
  timeout: 180000, // 3 minutes for cloud streaming tests
  expect: {
    timeout: 30000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
      },
    },
  ],
  // Output directory for test artifacts
  outputDir: "test-results/",
});
