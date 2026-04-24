# product-tests — product-level gate for Daydream Scope

This is a self-contained test system that treats **onboarding + stream-to-first-frame** as the #1 gate for every PR. Unlike `tests/` (which verifies code correctness) these tests exercise the full stack — real Scope subprocess, real browser, real WebRTC, real fal deployment for cloud — and treat "worked after a retry" as a **hard failure**, not a pass.

## Start here

| You are... | Read... |
|---|---|
| New to this system and want the tour | [`USER_GUIDE.md`](./USER_GUIDE.md) — shareable intro, how to run, how to read reports, how to participate |
| About to write a test | [`WRITING_TESTS.md`](./WRITING_TESTS.md) — cookbook, templates, ctx surface, testid map, gotchas |
| Writing a regression for a past bug | Ask Claude via the `/product-test-writer` skill, or copy [`_templates/regression.py.tpl`](./_templates/regression.py.tpl) |
| Doing a pre-release human sanity check | [`.agents/skills/onboarding-test/SKILL.md`](../.agents/skills/onboarding-test/SKILL.md) (Claude-in-Chrome plain-English walkthrough) |

## Directory layout

```
product-tests/
├── USER_GUIDE.md   — shareable system intro (start here if new)
├── WRITING_TESTS.md — cookbook for adding tests
├── _templates/     — copy-paste starting points (scenario, regression, chaos)
├── harness/        — reusable test plumbing (process mgmt, browser driver, observers, @scenario)
├── scenarios/      — happy-path product journeys
├── chaos/          — seeded chaotic-user simulations (rapid stop/start, parameter spam)
├── regression/     — one file per past bug, named after its PR number
├── release/        — slower, broader matrix run pre-tag (nightly ring)
├── contracts/      — cross-cutting invariants (no-retry, no-unexpected-close)
├── baselines/      — per-scenario latency/quality baselines for drift detection
└── reports/        — JSON + summary.md emission target (gitignored)
```

## Running locally

```bash
# one-time
uv sync --group product-tests
uv run playwright install chromium

# smoke
uv run pytest product-tests/scenarios/test_onboarding_local.py -v

# full scenario run
uv run pytest product-tests/scenarios/ -v

# chaos with reproducible seed
uv run pytest product-tests/chaos/ --chaos-seed=abc123

# cloud scenarios (requires SCOPE_CLOUD_APP_ID)
SCOPE_CLOUD_APP_ID=<pr-fal-app> uv run pytest product-tests/scenarios/test_onboarding_cloud.py
```

## What counts as a pass

Every test asserts **all** of:
- `retry_count == 0` across every instrumented counter
- `unexpected_close_count == 0` (session_closed not preceded by a test-initiated stop)
- `ui_error_events == 0` (no error toasts / stuck spinners past threshold)
- `first_frame_time_ms < baselines/<mode>.json[scenario]`
- `parameter_round_trip_ms_p95 < 500ms`

Any one of those failing = red. A successful first-frame after a retry is **not** a pass.

## CI rings

| Ring | Trigger | Budget | Pipelines | Runner |
|---|---|---|---|---|
| PR gate | Every PR + main/dev push | <25 min | `passthrough` only | ubuntu-latest CPU |
| Nightly | Cron + pre-release tag | <60 min | full models (longlive, ltx2) | GPU runner |

Both rings use **real** fal — PR gate via `deploy-PR-to-fal`, nightly against a pinned "latest main" fal app.
