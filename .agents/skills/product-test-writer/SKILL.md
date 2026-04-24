---
name: product-test-writer
description: Turn a plain-English bug description (or PR URL) into a runnable regression test under product-tests/regression/. Use when asked to write a regression, add a test for a past bug, reproduce an issue, or "add a product-test for #NNN".
---

# Product Test Writer

## What this skill does

You are given a plain-English description of a past bug (often: a PR number, a Linear issue, a Slack message). You produce **one file** at `product-tests/regression/test_pr_<NNN>_<slug>.py` that:

1. Documents the bug in its docstring (what the user did, what should have happened, what did, root cause, fix).
2. Uses the `@scenario` decorator — never raw fixtures.
3. Drives the reproduction via the `ctx` API.
4. Relies on the decorator's automatic gates for assertion (retries, unexpected closes, UI errors).

If the bug needs a different mode, different workflow, or a non-default timeout, say so in the code — not in a separate doc.

## Before writing anything

1. **Read the bug context.** If the user gave a PR number, `gh pr view <N>` it. If they gave a Linear ticket, ask them to paste the description. If they gave a brief sentence, ask 1–2 clarifying questions only if the mode/workflow/repro would be genuinely ambiguous.
2. **Read `product-tests/WRITING_TESTS.md`.** That's the source-of-truth for the `ctx` surface, testid map, and gotchas. It may have been updated since this skill was written.
3. **Grep for a similar existing test.** `product-tests/regression/` probably has one; `product-tests/scenarios/` might. If one already covers this failure mode, extend or dedupe — don't duplicate.

## The decision tree

| Question | If yes | If no |
|---|---|---|
| Does the bug only repro in cloud mode? | `mode="cloud"` | `mode="local"` (default; keeps PR ring fast) |
| Is it workflow-specific (a particular pipeline)? | `workflow="starter-..."` | `workflow="local-passthrough"` |
| Does it need chaotic timing to trigger? | Add `pytest.mark.chaos` and use `ctx.chaos()` | Linear reproduction in the body |
| Was the symptom a 5xx / crash? | Default gates catch it | Default gates catch it |
| Was the symptom silently-wrong output (no crash)? | Add an explicit assertion (e.g. compare `ctx.metrics()` or read a frame) | — |

## The template (copy this, then fill in)

```python
"""Regression for #<PR>: <one-line symptom>.

- What the user did:   <reproduction steps in plain English>
- What should happen:  <expected outcome>
- What did happen:     <observed symptom, incl. any log line patterns>
- Root cause:          <one-line root cause from the PR>
- Fix:                 <PR title / brief description of the fix>
"""

from __future__ import annotations

from harness.scenario import scenario


@scenario(mode="local", workflow="local-passthrough")
def test_pr_<PR>_<short_slug>(ctx):
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame()

    # -- reproduction --
    # Replace with the precise actions that reproduced the bug, using
    # ctx helpers (not raw page/driver) so stops are properly attributed.
    pass
```

## `ctx` surface you can use (memorize these, don't invent new ones)

| Action | Call |
|---|---|
| Onboard to graph view | `ctx.complete_onboarding()` |
| Run + wait first frame (records `first_frame_time_ms`) | `ctx.run_and_wait_first_frame(timeout_ms=60_000)` |
| Stop cleanly (marks + clicks, idempotent) | `ctx.stop_stream()` |
| Toggle Run/Stop without waiting | `ctx.toggle_run()` |
| Set a parameter over HTTP (returns status) | `ctx.set_parameter("name", value)` |
| Read current parameters | `ctx.get_parameters()` |
| Fetch session metrics | `ctx.metrics()` |
| Click/wait a `data-testid` | `ctx.click("testid")`, `ctx.wait("testid")` |
| Browser sleep (avoid unless you must) | `ctx.sleep(ms)` |
| Seeded chaos driver | `ctx.chaos()` |
| Record a dimension | `ctx.measure("name", value)` |
| Raw access when you must | `ctx.driver`, `ctx.page`, `ctx.base_url`, `ctx.retry_probe`, `ctx.failure_watcher`, `ctx.report` |

## Testid anchors (stable set; if you need one not listed, grep `frontend/src` for `data-testid`)

- `inference-mode-local`, `inference-mode-cloud`, `inference-mode-continue`
- `telemetry-accept`, `telemetry-decline`
- `workflow-card-<id>`, `workflow-get-started`, `workflow-import-load`
- `tour-next`, `tour-skip`
- `stream-run-stop` (attr `data-streaming="true"` when active)
- `sink-video`
- `cloud-toggle`

Workflow IDs: `local-passthrough` (CPU / PR-gate-safe), `starter-mythical-creature`, `starter-ref-image`, `starter-ltx-text-to-video` (GPU / nightly).

## Gotchas — do NOT violate these

1. **Never apply `@pytest.mark.cloud` manually.** Pass `mode="cloud"` to `@scenario`. The decorator applies the marker AND makes `ctx.complete_onboarding()` dispatch cloud.
2. **Never call `failure_watcher.mark_initiated_stop()` directly.** Use `ctx.stop_stream()` or `ctx.toggle_run()` — they handle it.
3. **Never call `gates.enforce_all_gates()` manually.** The decorator's teardown does it. Calling it twice is safe but signals you don't trust the contract — fix the root issue instead.
4. **Do not import raw fixtures (`scope_harness`, `driver`, `retry_probe`, etc.) in a new test.** If you think you need one, ask: can this use `ctx.<escape_hatch>` instead? Almost always yes.
5. **Do not reset retry counters mid-test** unless you're also going to write a comment explaining exactly why the warmup legitimately ticks them. Otherwise you're hiding evidence.
6. **File name must start with `test_`.** pytest collection rule.
7. **If the PR ring is CPU-only, the test must be too.** Use `local-passthrough` or a different PR-ring-safe workflow. GPU-specific bugs → nightly ring.

## Worked example

**Input:** "Add a regression for PR #1234 — users spamming the prompt slider during a cloud stream could crash the session. Fix was to debounce parameter updates."

**Output file:** `product-tests/regression/test_pr_1234_prompt_spam_during_cloud_stream.py`

```python
"""Regression for #1234: prompt spam during cloud stream crashed the session.

- What the user did:   On a running cloud stream, dragged the prompt slider
                       back and forth for ~10s (roughly 30–50 updates/sec).
- What should happen:  Each parameter update is accepted or coalesced; the
                       stream continues rendering.
- What did happen:     WebRTC data channel overflowed, session closed with
                       'forcibly closed' in scope.log, UI showed an error toast.
- Root cause:           Unbounded HTTP → data-channel fan-out in the parameter
                       broadcast path; backpressure was not enforced.
- Fix:                 Debounce + rate-limit parameter updates before
                       broadcasting (webrtc.py::broadcast_parameter_update).
"""

from __future__ import annotations

from harness.scenario import scenario


@scenario(mode="cloud", workflow="starter-mythical-creature")
def test_pr_1234_prompt_spam_during_cloud_stream(ctx):
    """Spam 200 parameter updates over HTTP; cloud session must survive."""
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame(timeout_ms=90_000)

    for i in range(200):
        ctx.set_parameter("__prompt", f"variant-{i}")

    # Give the pipeline a moment to process the tail of the spam.
    ctx.sleep(2000)

    # No explicit assertion needed. Decorator teardown will fail this test
    # if any retry fired, the session closed unexpectedly, or a UI error
    # toast appeared — which is exactly what happened pre-fix.
```

Notice what's NOT there: no fixture imports, no `failure_watcher.mark_initiated_stop()`, no `gates.enforce_all_gates()`, no `assert report.passed`. The decorator owns all of that.

## After writing

1. Run it: `uv run pytest product-tests/regression/test_pr_<NNN>_<slug>.py -v`. Report to the user whether it passed.
2. If the bug was not yet fixed on the current branch, expect it to **red**. That's correct — it proves the test actually reproduces the bug. Mention this to the user; they may want to gate the merge on this test.
3. If the test greens on an unfixed branch, the repro isn't tight enough — tighten it before landing.
4. Do NOT run `gh pr create` unless the user explicitly asks you to ship it.

## If the bug cannot be expressed in `ctx`

It's rare but real. Examples: the bug is in raw WebRTC negotiation (not covered by `ctx`); the bug only fires on a specific graph topology (needs a custom HTTP `session/start` body). In those cases:

1. Use `ctx.base_url` + raw `requests` for HTTP control-plane operations.
2. Use `ctx.page` for raw Playwright when a testid doesn't exist.
3. If you find yourself reaching for `ctx.failure_watcher` / `ctx.retry_probe` directly — stop. That's the decorator's job. If the decorator is in the way, the escape hatch is *not* to write a raw-fixture test; it's to improve `ctx` and re-target. File a note and ask.
