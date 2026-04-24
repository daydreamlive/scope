---
name: visual-qa
description: Triage a product-tests failure bundle. Reads the captured frames, screenshots, Playwright video, and Scope logs from a failed run and writes a plain-English "what went wrong here" summary.
---

# Visual QA — failure triage from captured artifacts

## When to use this

A `product-tests` run failed. You have a reports directory somewhere under
`product-tests/reports/<run_id>/<test>/` containing a mix of:

- JPEG frames from `ctx.capture_live_frame()`
- PNG screenshots from `ctx.screenshot()` / `ctx.screenshot_testid()`
- An MP4 `video.webm` from the Playwright recording
- `trace.zip` from the Playwright tracer
- `scope.log` from the backend subprocess
- `report.json` / `summary.md` from the test report
- Sometimes `triage.md` if the run had `SCOPE_MULTIMODAL_TRIAGE=1`

Your job is to produce a short, useful explanation of **what a human looking
at this would see** — the sort of thing that turns "here's trace.zip, good
luck" into "the workflow picker rendered 2 cards instead of 3 because its
container overflowed the right edge of the viewport."

This is the complement to `/product-test-writer`. That skill encodes a bug
description into a machine-runnable regression test. **This skill goes the
other direction**: a machine-runnable failure → a human-readable triage.

## Inputs

The user will give you a path like `product-tests/reports/20260423-143221/test_onboarding_local_passthrough/`. Open everything inside.

## What to produce

A single markdown file or chat response containing:

1. **TL;DR**: one sentence naming the visible symptom.
2. **Evidence**: list of specific artifacts you looked at and what each showed.
3. **Likely area**: one or two files in `frontend/src/` or `src/scope/server/` that most plausibly own the code path that produced the symptom. Pattern-match from the artifacts — a UI bug points at a component, a stream-output bug points at a pipeline or the recorder.
4. **Suggested next step**: either "run this test again with `SCOPE_MULTIMODAL_TRIAGE=1`" (if no triage.md existed), or "open these files to start", or "file a regression test via `/product-test-writer` with this description".

## Steps

1. **Read `summary.md`** first — it gives you the list of hard fails and the dimensions that tripped.
2. **Open `report.json`** — the `hard_fails` array, `dimensions`, and `metadata` fields. Specifically:
   - `dimensions.retry_count > 0` → the retry came from inference/relay/frontend-reconnects
   - `dimensions.unexpected_close_count > 0` → something closed the session behind the test's back
   - `dimensions.ui_error_events > 0` → an error toast fired in the browser
   - `dimensions.first_frame_time_ms > baseline` → slow first frame; check fal deployment logs
   - `metadata.multimodal_status == "fail"` → a visual assertion RED'd; `metadata.multimodal_reasoning` has the model's words
3. **View every image** in the directory (Claude Code's image-viewing capability — or the Chrome MCP if the user is driving you interactively). Describe what each shows in one line.
4. **Scan `scope.log`** for `ERROR`/`CRITICAL`/stack traces timestamped near the failure wall-clock.
5. **If `triage.md` exists, read it.** It's what the in-CI multimodal pass thought; quote the relevant bits.
6. **Pattern-match to a likely code area** using the reference map below.

## Reference: artifact → suspect-file map

| Symptom in artifacts | Likely owners |
|---|---|
| Modal cut off, tooltip misplaced, card missing | `frontend/src/components/onboarding/*`, `frontend/src/components/graph/*` |
| Run button stuck spinning, no frame landed | `src/scope/server/webrtc.py`, `src/scope/server/session.py` |
| Recording has zero frames or wrong FPS | `src/scope/server/recording/**`, pipeline's `__call__` for PTS behavior |
| Sink output all-black / all-one-color | pipeline `__call__` in `src/scope/core/pipelines/<name>/` |
| Error toast "Cloud unavailable" / retries > 0 | `src/scope/server/livepeer.py`, `src/scope/server/cloud_relay.py`, `frontend/src/hooks/useUnifiedWebRTC.ts` |
| Onboarding re-triggers on restart | `frontend/src/contexts/OnboardingContext.tsx`, `src/scope/server/app.py` (onboarding.json I/O) |
| Tour popover pointing at nothing | `frontend/src/components/onboarding/TourPopover.tsx` |

## Output template

```markdown
# Triage — <test_name>

## TL;DR
<one sentence naming the visible symptom>

## Evidence
- `<filename>`: <what you saw>
- `<filename>`: <what you saw>
- `scope.log` @ <timestamp>: <relevant line>

## Likely area
<file path(s)>

## Next step
<one of: rerun with SCOPE_MULTIMODAL_TRIAGE=1, open these files, file a regression>
```

## Limits

- **Do NOT speculate past the artifacts.** "I can't tell from the screenshots whether the click registered" is valid output. Guessing is not.
- **Don't propose a fix.** This skill is diagnostic. If you know what needs to change, say so in "Next step" but do not edit code.
- **Don't run the test again** unless the user asks. You're reading a post-mortem, not reproducing.
