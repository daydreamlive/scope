---
name: onboarding-test
description: RETIRED — superseded by product-tests/. Use those scenarios instead of Claude-in-Chrome automation.
---

# Onboarding Test — Retired

This Claude-in-Chrome-driven onboarding test has been replaced by the
self-contained Python/Playwright product-tests system.

## Where to go instead

- **Run the local onboarding scenario locally:**

  ```bash
  uv sync --group product-tests
  uv run playwright install chromium
  cd product-tests && uv run pytest scenarios/test_onboarding_local.py
  ```

- **Run the cloud onboarding scenario:**

  ```bash
  SCOPE_CLOUD_APP_ID=daydream/scope-livepeer/ws \
    uv run pytest product-tests/scenarios/test_onboarding_cloud.py
  ```

- **CI coverage:** `.github/workflows/product-tests.yml` runs the PR gate
  on every push and a nightly with GPU + full models.

## Why it was retired

The old skill drove a real Chrome browser through Claude's MCP tools and
had no way to:

1. Count retries as hard failures (flaky/"eventually worked" runs passed).
2. Detect unexpected session closes that happen silently in logs.
3. Simulate chaotic user behavior with reproducible seeds.
4. Gate PRs — it ran only when Claude was asked to run it.

The new system (see `product-tests/README.md`) treats the onboarding
workflows on both local and cloud mode as the #1 gate and runs them on
every PR.

## Source of truth for the old flow

The old skill's step-by-step click map lives in git history; the
product-tests equivalent is in
[product-tests/harness/flows.py](../../../product-tests/harness/flows.py)
in the `complete_onboarding_local` and `complete_onboarding_cloud`
helpers.
