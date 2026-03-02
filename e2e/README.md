# Scope E2E Tests

End-to-end tests for Scope cloud streaming via fal.ai.

## Overview

These tests verify the full cloud streaming flow:
1. Login to Daydream web app
2. Connect to cloud mode (fal deployment)
3. Start a stream with the passthrough model
4. Verify frames are being processed
5. Stop stream

## Prerequisites

- Node.js 22+
- A Daydream test account
- A fal deployment to test against

## Setup

```bash
cd e2e
npm install
npx playwright install --with-deps chromium
```

## Running Tests

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FAL_WS_URL` | Yes | WebSocket URL for the fal deployment (e.g., `wss://fal.run/livepeer/scope-pr-123/ws`) |
| `DAYDREAM_TEST_EMAIL` | Yes | Test user email for Daydream login |
| `DAYDREAM_TEST_PASSWORD` | Yes | Test user password |
| `DAYDREAM_BASE_URL` | No | Base URL for Daydream app (default: `https://app.daydream.live`) |

### Run Tests

```bash
# Headless mode (CI)
FAL_WS_URL=wss://fal.run/livepeer/scope/ws \
DAYDREAM_TEST_EMAIL=test@example.com \
DAYDREAM_TEST_PASSWORD=secret \
npm test

# With browser visible (debugging)
npm run test:headed

# Interactive UI mode
npm run test:ui

# Debug mode (step through)
npm run test:debug
```

### View Report

After running tests:

```bash
npm run report
```

## CI Integration

These tests run automatically on every PR via GitHub Actions:

1. **Docker Build** workflow builds the image
2. **Deploy PR to fal** workflow deploys to a PR-specific fal app
3. **E2E Tests** workflow runs these tests against the deployment

Results are posted as comments on the PR.

## Test Structure

```
e2e/
├── tests/
│   ├── auth.setup.ts       # Authentication setup (runs first)
│   └── cloud-streaming.spec.ts  # Main cloud streaming test
├── playwright.config.ts    # Playwright configuration
├── package.json
└── README.md
```

## Debugging Failed Tests

When tests fail in CI:
1. Check the workflow run for logs
2. Download the `test-artifacts` artifact for:
   - Screenshots on failure
   - Video recordings
   - Playwright traces

To view traces locally:
```bash
npx playwright show-trace path/to/trace.zip
```

## Writing New Tests

```typescript
import { test, expect } from "@playwright/test";

test("my new cloud test", async ({ page }) => {
  // Tests use saved auth state, so you're already logged in
  await page.goto("/");
  
  // Your test logic here
  // Use data-testid attributes for reliable selectors
  const element = page.locator('[data-testid="my-element"]');
  await expect(element).toBeVisible();
});
```
