# Scope E2E Tests

<<<<<<< HEAD
End-to-end tests for Scope onboarding and Livepeer cloud workflows.
=======
End-to-end Playwright test for Scope's Livepeer cloud streaming path.
>>>>>>> 6a177ad5 (docs: make the testing-livepeer-fal-deploy skill discoverable)

## What it verifies

<<<<<<< HEAD
These tests verify the full cloud flow:
1. Login to Daydream web app
2. Connect to Livepeer cloud mode
3. Start a stream with the passthrough model
4. Verify frames are being processed
5. Stop stream
=======
The single test in `tests/cloud-streaming.spec.ts` drives the full
round-trip via a real browser:
>>>>>>> 6a177ad5 (docs: make the testing-livepeer-fal-deploy skill discoverable)

1. App loads (signed-in via a baked-in API key)
2. Switch to Perform mode
3. Toggle Remote Inference on, wait for cloud connection
4. Select the `passthrough` pipeline
5. Switch input to Camera (headless Chromium gets a synthetic feed)
6. Start the stream
7. Verify the **output** `<video>` in the "Video Output" card is
   actually playing (frames round-tripped through the fal runner)
8. Stop the stream

<<<<<<< HEAD
- Node.js 22+
- A Daydream test account
- A deployed Livepeer runner to test against
=======
## For the full setup guide
>>>>>>> 6a177ad5 (docs: make the testing-livepeer-fal-deploy skill discoverable)

This directory is intentionally minimal. The canonical setup and
workflow instructions — including `.env.local` contents, sudo system
deps for Chromium (`libnss3 libnspr4 libasound2t64`), expected
Kafka/ClickHouse event sequence, and common failure signatures — live
in the Claude Code skill:

```
.agents/skills/testing-livepeer-fal-deploy/SKILL.md
```

Ask Claude to "test the fal deploy" (or any other trigger phrase from
the skill's `description`) and it will walk the flow. Or read the
SKILL.md directly.

## Quick reference

```bash
# One-time setup
cd e2e
npm install
npx playwright install chromium
sudo apt-get install -y libnss3 libnspr4 libasound2t64  # first time only

# Bake the API key into the frontend
source ../.env.local
(cd ../frontend && VITE_DAYDREAM_API_KEY="$SCOPE_CLOUD_API_KEY" npm run build)

# Run
../run-app.sh &           # scope on :8000
npx playwright test       # ~2–5 min

# Debug variants
npm run test:headed       # visible browser
npm run test:ui           # interactive UI
npm run test:debug        # step through
npm run report            # open last HTML report
```

## Env vars (via `.env.local`)

See `.env.example` at the repo root. Required: `SCOPE_CLOUD_APP_ID`,
`SCOPE_CLOUD_API_KEY`, `SCOPE_USER_ID`. Optional: `LIVEPEER_DEBUG=1`.

<<<<<<< HEAD
| Variable | Required | Description |
|----------|----------|-------------|
| `SCOPE_CLOUD_APP_ID` | Yes | Livepeer fal app ID (e.g., `daydream/scope-livepeer-pr-123--preview/ws`) |
| `DAYDREAM_TEST_EMAIL` | Yes | Test user email for Daydream login |
| `DAYDREAM_TEST_PASSWORD` | Yes | Test user password |
| `DAYDREAM_BASE_URL` | No | Base URL for Daydream app (default: `https://app.daydream.live`) |
=======
## Fast HTTP-only smoke (no browser)
>>>>>>> 6a177ad5 (docs: make the testing-livepeer-fal-deploy skill discoverable)

For a quick "did the fal container come up?" check — bisect-friendly,
no Playwright needed:

```bash
<<<<<<< HEAD
# Headless mode (CI)
SCOPE_CLOUD_APP_ID=daydream/scope-livepeer--prod/ws \
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
2. **Deploy PR to fal** workflow deploys a PR-specific Livepeer runner
3. **E2E Tests** workflow runs these tests against the deployment

Results are posted as comments on the PR.

## Test Structure

```
e2e/
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
=======
../test-cloud-connect.sh --skip-push --skip-build-wait --skip-deploy
```

This only exercises `/api/v1/cloud/connect`; it will not produce the
`pipeline_loaded` / `session_created` / `stream_started` Kafka events
that the Playwright test does. Use it for infrastructure-level
regressions; use Playwright for everything else.
>>>>>>> 6a177ad5 (docs: make the testing-livepeer-fal-deploy skill discoverable)
