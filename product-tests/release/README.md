# Release-gate scenarios

This directory holds the nightly / pre-release matrix tests: the deep,
long-running coverage that runs on GPU runners against the latest-main
fal app. Think of it as the "ship/no-ship" gate that complements the
fast PR gate in `../scenarios/`.

Tests in this directory:

- Run **only** in the nightly ring (see `.github/workflows/product-tests.yml`).
- Exercise the full model matrix (LongLive, LTX, etc.), not just CPU pipelines.
- May take tens of minutes per scenario.
- Share fixtures and gates with the rest of `product-tests/` (same conftest,
  same contract enforcement).

This supersedes the retired `e2e/` TypeScript Playwright scaffold at the
repo root. Full-model coverage that used to live there now lives here.
