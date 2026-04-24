# Daydream Scope — product-tests user guide

A shareable walkthrough of the test system that gates every PR on product quality, not just code correctness. Written for engineers, designers, and PMs who want to understand what the suite does, how to run it, how to read the results, and how to contribute.

> **TL;DR** — `product-tests/` runs real UI in a real browser against a real Scope subprocess and fails a PR the moment a retry fires, a session closes unexpectedly, or first-frame time drifts past baseline. If you've ever shipped a change that was "green in CI, broken for the first user who tried it," this system exists to make that harder.

---

## 1. Why this exists

Unit tests answer "does the code work in isolation?" They can't answer the questions users actually care about:

- **Did it take 3 retries to get a video frame?** That's a pass for the old suite. It's a fail here.
- **Did the session forcibly close and silently reconnect?** Same.
- **Did the UI feel sluggish? Show an error toast?** Same.

The three failure modes that actually break Daydream Scope in the wild are:

1. **Remote inference failing** (fal-hosted model crashes, times out, or returns garbage)
2. **Scope ↔ remote-inference bad interactions** — session torn down and brought back up quickly, state clashes, forcibly-closed WebRTC sessions
3. **UI bugs** that make the product *look* broken even when the backend is fine

`product-tests/` exercises all three systematically, on every PR.

## 2. What it does (in plain English)

For every PR, the suite:

1. Boots a fresh Scope subprocess with isolated storage and a retry counter enabled.
2. Launches a real Chromium browser via Playwright, navigates to the Scope URL.
3. Drives onboarding — picks a provider, declines telemetry, selects a workflow, dismisses tour tooltips, clicks Run.
4. Waits for the first video frame to render in the sink `<video>` element.
5. Simulates chaotic user behavior in parallel tests — rapid stop/run toggles, parameter spam, workflow switching, tab backgrounding, graph mutation.
6. At teardown, asserts:
   - **Zero retries** across every instrumented counter (server reconnects, FE reconnects, frame drops).
   - **Zero unexpected session closes** (a `session_closed` event not preceded by a test-initiated stop = fail).
   - **Zero UI error events** (no error toasts, no stuck spinners).
   - **First-frame time within baseline** (`baselines/local.json`, `baselines/cloud.json`).
7. Emits a per-test `report.json` + a run-level `summary.md` that's posted as a PR comment.

The key invariant: **"worked after a retry" is not a pass.** If your change makes Scope transiently unhealthy in a way the old suite would have tolerated, this suite will catch it.

## 3. Two test surfaces, two audiences

We have two complementary ways to test. Pick the right one:

| You want to answer... | Use... |
|---|---|
| "Is this correct, and is it regressing product quality?" (every PR, machine-readable) | `product-tests/` — this suite. pytest + Playwright + retry-counter gates. |
| "Does it *feel* right?" (pre-release sanity pass, eyeballs-on, institutional knowledge in plain English) | `.agents/skills/onboarding-test/` — a Claude-in-Chrome skill that drives the UI and reports in English. |

Both are kept, because they catch different things. The automated suite won't tell you "the loading spinner sat there motionless for 45s and that felt broken"; the human walkthrough won't give you deterministic replay and a merge gate. Use both.

## 4. The architecture, at 100 feet

```
┌───────────────────────────── pytest session ─────────────────────────────┐
│                                                                          │
│   ┌──────────────────────── per-test fixtures ────────────────────────┐  │
│   │                                                                    │  │
│   │  scope_harness   → spawns fresh `uv run daydream-scope --port N`  │  │
│   │      │              with isolated DAYDREAM_SCOPE_DIR + instrument │  │
│   │      │                                                             │  │
│   │      ├─ driver          (Playwright Chromium context)              │  │
│   │      ├─ retry_probe     (reads /api/v1/_debug/retry_stats)         │  │
│   │      ├─ failure_watcher (tails scope.log for unexpected closes)    │  │
│   │      └─ report          (per-test JSON emission)                   │  │
│   │                                                                    │  │
│   └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│   @scenario(mode="local", workflow="local-passthrough")                   │
│   def test_my_thing(ctx):                                                │
│       ctx.complete_onboarding()                                          │
│       ctx.run_and_wait_first_frame()                                     │
│       # ... reproduction steps ...                                       │
│       # teardown auto-asserts: zero retries, zero unexpected closes,     │
│       # zero UI errors, stream stopped cleanly.                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

The `@scenario` decorator hides all fixture plumbing. A test body becomes "what the user does"; the decorator handles "what the harness checks."

## 5. Running locally

One-time setup:

```bash
uv sync --group product-tests
uv run playwright install chromium
```

Run the onboarding smoke test:

```bash
uv run pytest product-tests/scenarios/test_onboarding_local.py -v
```

Run everything the PR gate runs:

```bash
uv run pytest product-tests/scenarios/ product-tests/chaos/ -v -m "not cloud and not slow"
```

Reproducible chaos run (same seed ⇒ same timeline):

```bash
uv run pytest product-tests/chaos/ --chaos-seed=abc123 -v
```

Cloud tests (requires a deployed fal app):

```bash
SCOPE_CLOUD_APP_ID=daydream/scope-livepeer-pr-NNN--preview/ws \
  uv run pytest product-tests/scenarios/test_onboarding_cloud.py -v
```

Nightly full matrix (GPU, all three starter workflows):

```bash
SCOPE_CLOUD_RING=nightly \
SCOPE_CLOUD_APP_ID=daydream/scope-livepeer--prod/ws \
  uv run pytest product-tests/release/ -v
```

## 6. Reading the reports

Every run writes to `product-tests/reports/<run-id>/<test-id>/`:

| File | Contains |
|---|---|
| `report.json` | Dimensions, pass/fail, hard-fail reasons, metadata, artifact paths |
| `scope.log` | Full stderr/stdout of the Scope subprocess that test booted |
| `video.webm` | Playwright video recording of the entire test |
| `trace.zip` | Playwright trace — open with `playwright show-trace trace.zip` |
| `timeline.jsonl` | Chaos action timeline, one JSON object per injected action |

A run-level `summary.md` rolls up every test into a PR-comment-friendly markdown table.

### The summary table, explained

```
| test | mode | pass | first_frame_ms | drift | retries | unexpected_closes |
|---|---|---|---|---|---|---|
| test_onboarding_local_passthrough | local | ✅ | 8420 | -3.2% | 0 | 0 |
```

- **first_frame_ms** — raw measurement. Lower is better.
- **drift** — percentage vs. the baseline in `baselines/<mode>.json`. Negative = faster than baseline (good). Positive = slower (investigate). A `—` means no baseline is recorded for this scenario yet.
- **retries**, **unexpected_closes** — must both be `0`. Anything else is a red.

Hard failures (if any) are listed below the table with the specific reason.

## 7. How to add a test (three ways)

### Option A — Ask Claude

Have a bug report? Describe it to Claude, optionally with a PR link or Linear ticket:

> *"Write a product-test for PR #1234 — users spamming the prompt slider during a cloud stream could crash the session. The fix debounces parameter updates."*

The `/product-test-writer` skill produces `product-tests/regression/test_pr_1234_<slug>.py` using the `@scenario` template, with the right mode, workflow, and testid anchors. Review, run, merge.

### Option B — Copy a template

```bash
cp product-tests/_templates/regression.py.tpl \
   product-tests/regression/test_pr_1234_parameter_spam.py
$EDITOR product-tests/regression/test_pr_1234_parameter_spam.py
```

Fill in the docstring, the two or three lines of reproduction, and you're done. See [WRITING_TESTS.md](./WRITING_TESTS.md) for the full cookbook.

### Option C — Read a reference implementation

- Minimal scenario: [`scenarios/test_onboarding_local.py`](./scenarios/test_onboarding_local.py) (15 lines, gates first-frame SLO).
- Chaos: [`chaos/test_rapid_stop_start.py`](./chaos/test_rapid_stop_start.py) (seeded random toggles, asserts every Run produces a frame).
- Regression skeleton: [`_templates/regression.py.tpl`](./_templates/regression.py.tpl).

## 8. The `@scenario` API (the 80% you need)

```python
from harness.scenario import scenario

@scenario(mode="local", workflow="local-passthrough")
def test_my_thing(ctx):
    ctx.complete_onboarding()             # walks onboarding, lands on graph view
    ctx.run_and_wait_first_frame()        # clicks Run, waits for first frame
    ctx.set_parameter("__prompt", "hi")   # POST /api/v1/session/parameters
    metrics = ctx.metrics()               # GET /api/v1/session/metrics
    ctx.stop_stream()                     # mark initiated stop + click Stop
```

Full reference: [WRITING_TESTS.md § ctx surface](./WRITING_TESTS.md#ctx-surface).

## 9. CI — what happens on a PR

Two rings, both wired in `.github/workflows/product-tests.yml`:

| Ring | Trigger | Budget | Workflows | Runner |
|---|---|---|---|---|
| **PR gate** | Every PR + push to main/dev | < 25 min | `local-passthrough` (CPU) + one cloud smoke | `ubuntu-latest` |
| **Nightly** | Cron + pre-release tag | < 60 min | Full starter matrix (`mythical-creature`, `ref-image`, `ltx-text-to-video`) | GPU self-hosted |

On every PR you get:

- **A PR comment** with the summary table (passed/failed, first-frame times, baseline drift). Updates on every push via `marocchino/sticky-pull-request-comment@v2`.
- **Artifact upload on failure** — the full `reports/<run-id>/` tree, including videos and traces, for 14 days.
- **Merge blocked** if any test reds.

Cloud smoke on the PR ring points at a PR-specific fal deployment (via the existing `deploy-PR-to-fal` workflow). Nightly points at a pinned `latest main` fal app.

## 10. How to participate

- **Hit a bug in development?** Write a regression test for it with the `/product-test-writer` skill, or copy `_templates/regression.py.tpl`. One file per bug, named `test_pr_<NNN>_<slug>.py`. Having a deterministic repro pinned to the PR number makes triage later dramatically easier.
- **PR comment shows drift on your change?** Drift doesn't fail the test by itself, but +15% on `first_frame_time_ms` for three PRs in a row is how we wake up to a slow regression. Look at the trace, figure out where the time went.
- **A test is flaky?** File it as a bug, not a "rerun." A flaky product-test usually means there's a real race condition; re-running masks it. Post the seed from `--chaos-seed=` and the `report.json` to the thread.
- **Need a testid that doesn't exist?** Add it to the component (`data-testid="..."`) and document it in [WRITING_TESTS.md](./WRITING_TESTS.md#data-testid-map). Don't select by text content or CSS classes — those break the moment someone does a visual polish pass.
- **Writing a new workflow / pipeline?** Add a baseline entry in `baselines/local.json` or `baselines/cloud.json` with the first-frame SLO you think is reasonable. Missing baselines default to effectively-infinite so new workflows don't silently pass.
- **Want the human eyeballs-on sanity check before a release?** Use `.agents/skills/onboarding-test/` — walks you (or Claude-in-Chrome) through the onboarding flows with a plain-English checklist.

## 11. Glossary

| Term | Meaning |
|---|---|
| **Retry** | Any reconnect/retry that fires in the server, frontend, or cloud relay. Instrumented in `src/scope/server/retry_counter.py`. Any retry = test red. |
| **Unexpected close** | `session_closed` event or log line without a preceding `failure_watcher.mark_initiated_stop()`. |
| **Gate** | A teardown-time assertion that runs regardless of test outcome. Defined in `harness/gates.py`. |
| **SLO** | First-frame time budget per `(mode, workflow)`. Defined in `baselines/<mode>.json`. |
| **Ring** | PR gate (runs on every PR, 25 min, CPU) vs. nightly (GPU, full matrix, 60 min). |
| **Chaos seed** | The seed string that makes a chaos test byte-reproducible. Defaults to the git SHA; override with `--chaos-seed=`. |
| **`ctx`** | The high-level test API. A `ScenarioContext` instance that bundles driver + harness + report. |

## 12. The Chrome-MCP → regression-test loop

This is the *end-to-end* bug-to-test flow that makes the system "as capable as a
human QA pass." It stitches three skills together:

1. **A human (or Claude) finds a UI/visual bug** — either during development,
   a PR review walkthrough, or a pre-release sanity check — by using
   [`.agents/skills/onboarding-test`](../.agents/skills/onboarding-test/SKILL.md).
   That skill drives Chrome via MCP and the user sees the problem directly.
2. **The reviewer describes the bug in plain English** — "the third workflow
   card is clipped on a 1440px viewport", "the tour popover is pointing at
   empty space", "the recorded MP4 stutters".
3. **`/product-test-writer` converts the description into a running test** —
   writes a file under `product-tests/regression/` that uses
   `ctx.screenshot_testid(...)` + `ctx.multimodal_check(...)` (or a cheaper
   pixel-stat assertion when applicable), runs it, shows it red on `main`,
   you fix the bug, it goes green. That's the round-trip.
4. **If a later CI run fails unexpectedly** — point
   [`.agents/skills/visual-qa`](../.agents/skills/visual-qa/SKILL.md) at the
   reports directory. It reads the captured frames, screenshots, Playwright
   video, and `scope.log`, and writes a plain-English triage summary. Pairs
   well with an `SCOPE_MULTIMODAL_TRIAGE=1` in-CI pass that already left a
   `triage.md` in the report dir.

### Opting into multimodal locally

```bash
# One-time — add ANTHROPIC_API_KEY to your env.
export ANTHROPIC_API_KEY="sk-ant-..."

# Run just the multimodal tests locally.
SCOPE_MULTIMODAL_EVAL=1 \
  uv run pytest product-tests/ -m multimodal -v

# Run everything with on-failure triage writing a triage.md for any red test.
SCOPE_MULTIMODAL_EVAL=1 SCOPE_MULTIMODAL_TRIAGE=1 \
  uv run pytest product-tests/scenarios/ -v

# Cap the daily spend (calls past the cap return "uncertain", don't red).
export SCOPE_MULTIMODAL_BUDGET_USD=5.00
```

Multimodal is opt-in, default-off. Without `SCOPE_MULTIMODAL_EVAL=1`, the
`@pytest.mark.multimodal` tests still run and capture artifacts — they just
return an "uncertain" verdict and skip the assertion, so local dev doesn't
accidentally burn API credit. The nightly CI ring runs multimodal with the
team's shared key; the PR ring runs them only when a PR touches
`frontend/src/components/onboarding/**` or `frontend/src/components/graph/**`.

### Where the three skills fit

| Skill | When | What it does |
|---|---|---|
| `onboarding-test` | Human wants to feel the product; pre-release sanity | Drives Chrome via MCP, plain-English walkthrough, visual verification |
| `product-test-writer` | You found a bug you want to prevent recurring | Turns the description into a `@scenario` regression test |
| `visual-qa` | A CI run failed and you want to know what a human would see | Reads the reports bundle; writes a triage summary |

## 13. Further reading

- [`WRITING_TESTS.md`](./WRITING_TESTS.md) — the cookbook. Templates, ctx surface, testid map, gotchas.
- [`README.md`](./README.md) — one-screen summary and pass criteria.
- [`_templates/`](./_templates/) — fillable starting points for scenario / regression / chaos tests.
- [`.agents/skills/product-test-writer/SKILL.md`](../.agents/skills/product-test-writer/SKILL.md) — Claude skill that writes regressions from plain-English bug descriptions.
- [`.agents/skills/onboarding-test/SKILL.md`](../.agents/skills/onboarding-test/SKILL.md) — Claude-in-Chrome skill for the human eyeballs-on walkthrough.
- [`.agents/skills/visual-qa/SKILL.md`](../.agents/skills/visual-qa/SKILL.md) — Claude skill for triaging a failure bundle into a plain-English summary.
- `harness/media.py` — ffprobe + SSIM + perceptual hashing helpers for media-quality assertions.
- `harness/visual_eval.py` — Anthropic vision wrapper with budget + caching.
- `harness/testids.py` — generated constants for every frontend `data-testid`. Regenerate with `uv run python -m harness.testids --sync`.
- `.github/workflows/product-tests.yml` — CI wiring.

## 14. FAQ

**"Why Python tests on a TypeScript frontend?"**
Because the harness needs to spawn + supervise a Scope subprocess, subscribe to the `/api/v1/events` WebSocket, tail logs, and call HTTP APIs. That work lives next to the Python server it's testing. Playwright's sync Python API drives Chromium just fine.

**"Can I run these against my local dev Scope instead of a fresh subprocess?"**
No — intentionally. The gate's whole point is "a fresh user, a fresh install, zero retries." Reusing a warm Scope hides the cold-start issues that bite first-time users.

**"My test retries once on startup and it's hard to avoid. Can I reset the counter after warmup?"**
You *can* (`ctx.retry_probe.reset()`), but think twice. If a warmup retry is legitimate, the gate is telling you the cold-start path has a real issue that a first-time user will see. Fixing the root cause beats hiding it. If you truly must reset, leave a comment explaining why.

**"Can I parametrize a `@scenario` test?"**
Not directly — the decorator wraps the body to inject the five fixtures. For parametric tests, use the raw fixture signature (see `release/test_cloud_full_matrix.py` for the pattern). Most regression tests don't need parametrization; one file per bug is fine.

**"My PR only touches docs / a linter config — do I need the suite to run?"**
The PR gate runs on every PR, but finishes in under a minute when nothing under `src/scope/server/` or `frontend/src/` changed (the scope-harness subprocess still boots to prove Scope didn't break). There's no opt-out; if it's noisy on a pure-docs PR, file that as a CI issue.

**"I broke a baseline. Now what?"**
If the drift is legitimate (you moved the SLO on purpose), update `baselines/<mode>.json` in the same PR and mention it in the description. If the drift is accidental, find the regression — the trace file will usually point you at the slow step.

---

Questions, confusion, or a gotcha that cost you an afternoon? File it as an edit to this file. This guide is the source of truth for how the system works; if it's wrong, fix it here.
