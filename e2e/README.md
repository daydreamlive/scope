# e2e/ — RETIRED

This TypeScript Playwright scaffold has been superseded by the Python
product-tests system at [`../product-tests/`](../product-tests/README.md).

## Where to go instead

- **PR-gate cloud smoke:** `product-tests/scenarios/test_onboarding_cloud.py`
- **Nightly full-matrix cloud:** `product-tests/release/test_cloud_full_matrix.py`
- **CI wiring:** `.github/workflows/product-tests.yml`

## Why it was retired

The old scaffold had TypeScript + `@playwright/test` infrastructure but
no actual test bodies, no retry-counter gating, no chaos simulation, and
no PR-comment integration. The new system treats onboarding (local +
cloud) as the #1 gate, counts retries/unexpected closes as hard fails,
and scores runs across multiple product-quality dimensions.

## Running the migrated tests

```bash
# Install the product-tests dep group:
uv sync --group product-tests
uv run playwright install chromium

# Local PR gate:
cd product-tests && uv run pytest scenarios/ chaos/

# Cloud (PR-deployed fal app):
SCOPE_CLOUD_APP_ID=daydream/scope-livepeer-pr-123--preview/ws \
  uv run pytest product-tests/scenarios/test_onboarding_cloud.py

# Nightly full matrix:
SCOPE_CLOUD_RING=nightly \
SCOPE_CLOUD_APP_ID=daydream/scope-livepeer--prod/ws \
  uv run pytest product-tests/release/
```

## Leftover files

`package.json`, `package-lock.json`, and `playwright.config.ts` remain
in place to avoid breaking any in-flight CI references. They can be
removed in a follow-up cleanup PR once the product-tests CI rings have
run green for a cycle.
