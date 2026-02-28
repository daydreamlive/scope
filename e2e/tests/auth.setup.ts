import { test as setup, expect } from "@playwright/test";
import path from "path";

const authFile = path.join(__dirname, "../playwright/.auth/user.json");

/**
 * Authentication setup for Daydream.
 *
 * This logs into the Daydream web app and saves the auth state
 * for use by other tests.
 *
 * Required environment variables:
 * - DAYDREAM_TEST_EMAIL: Test user email
 * - DAYDREAM_TEST_PASSWORD: Test user password
 */
setup("authenticate", async ({ page }) => {
  const email = process.env.DAYDREAM_TEST_EMAIL;
  const password = process.env.DAYDREAM_TEST_PASSWORD;

  if (!email || !password) {
    throw new Error(
      "DAYDREAM_TEST_EMAIL and DAYDREAM_TEST_PASSWORD must be set"
    );
  }

  // Go to Daydream app
  await page.goto("/");

  // Wait for page to load
  await page.waitForLoadState("networkidle");

  // Check if we need to log in
  // The app might redirect to login or show a login button
  const loginButton = page.getByRole("button", { name: /sign in|log in/i });
  const isLoggedIn = await page
    .getByRole("button", { name: /create|stream/i })
    .isVisible()
    .catch(() => false);

  if (!isLoggedIn) {
    // Click login if there's a button
    if (await loginButton.isVisible()) {
      await loginButton.click();
    }

    // Wait for login form/modal
    // Daydream likely uses a third-party auth provider (Google, GitHub, etc.)
    // or email/password. We'll try to handle common patterns.

    // Try email input
    const emailInput = page.getByRole("textbox", { name: /email/i });
    if (await emailInput.isVisible({ timeout: 5000 }).catch(() => false)) {
      await emailInput.fill(email);

      // Look for password field
      const passwordInput = page.getByRole("textbox", { name: /password/i });
      if (await passwordInput.isVisible().catch(() => false)) {
        await passwordInput.fill(password);
      }

      // Submit form
      const submitButton = page.getByRole("button", {
        name: /sign in|log in|continue/i,
      });
      await submitButton.click();
    }

    // Wait for redirect back to app after login
    await page.waitForURL("**/", { timeout: 30000 });
  }

  // Verify we're logged in
  await expect(
    page.getByRole("button", { name: /create|stream|profile/i })
  ).toBeVisible({ timeout: 10000 });

  // Save auth state
  await page.context().storageState({ path: authFile });

  console.log("✅ Authentication successful, state saved");
});
