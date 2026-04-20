# Scope E2E Tests

End-to-end Playwright test for Scope's Livepeer cloud streaming path.

## What it verifies

The single test in `tests/cloud-streaming.spec.ts` drives the full
round-trip via a real browser:

1. App loads (signed-in via a baked-in API key)
2. Switch to Perform mode
3. Toggle Remote Inference on, wait for cloud connection
4. Select the `passthrough` pipeline
5. Switch input to Camera (headless Chromium gets a synthetic feed)
6. Start the stream
7. Verify the **output** `<video>` in the "Video Output" card is
   actually playing (frames round-tripped through the fal runner)
8. Stop the stream

## For the full setup guide

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

## Fast HTTP-only smoke (no browser)

For a quick "did the fal container come up?" check — bisect-friendly,
no Playwright needed:

```bash
../test-cloud-connect.sh --skip-push --skip-build-wait --skip-deploy
```

This only exercises `/api/v1/cloud/connect`; it will not produce the
`pipeline_loaded` / `session_created` / `stream_started` Kafka events
that the Playwright test does. Use it for infrastructure-level
regressions; use Playwright for everything else.
