import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for Scope E2E tests.
 *
 * Environment variables:
 * - DAYDREAM_BASE_URL: Base URL for the Daydream web app (default: https://app.daydream.live)
 * - FAL_WS_URL: WebSocket URL for the fal deployment to test
 * - DAYDREAM_TEST_EMAIL: Test user email
 * - DAYDREAM_TEST_PASSWORD: Test user password
 */
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false, // Run tests serially for now (shared auth state)
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: [
    ["html", { open: "never" }],
    ["list"],
    ...(process.env.CI ? [["github" as const]] : []),
  ],
  use: {
    baseURL: process.env.DAYDREAM_BASE_URL || "https://app.daydream.live",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
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
    // Setup project for authentication
    {
      name: "setup",
      testMatch: /.*\.setup\.ts/,
    },
    // Main test project
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        // Use saved auth state
        storageState: "playwright/.auth/user.json",
      },
      dependencies: ["setup"],
    },
  ],
  // Output directory for test artifacts
  outputDir: "test-results/",
});
