# Writing a product-test

**Target: 15 minutes, copy-paste a template, ship a passing regression test.**

If anything below is out of date or you hit a gotcha not listed, fix it here — this file is the source of truth.

## TL;DR

1. Copy a template from [`_templates/`](./_templates) into the right folder.
2. Fill in 2–5 lines.
3. Run it. Merge it.

```python
# product-tests/regression/pr_1234_parameter_spam.py
"""Regression for #1234: parameter spam crashed the session."""
from harness.scenario import scenario

@scenario(mode="local", workflow="local-passthrough")
def test_pr_1234_parameter_spam(ctx):
    ctx.complete_onboarding()
    ctx.run_and_wait_first_frame()
    for _ in range(200):
        ctx.set_parameter("__prompt", "hello")
    # teardown auto-asserts: zero retries, zero unexpected closes,
    # zero UI errors, stream stopped cleanly.
```

## Where does my test go?

| Folder | Use for | Naming |
|---|---|---|
| `scenarios/` | Happy-path product journeys that must stay green on every PR. | `test_<feature>.py` |
| `regression/` | One file per past bug. The file itself documents the bug. | `test_pr_<NNN>_<slug>.py` |
| `chaos/` | Seeded chaotic user simulations (rapid toggles, parameter spam, etc.). | `test_<chaos_mode>.py` |
| `release/` | Slower, broader matrix run pre-tag. | `test_<matrix>.py` |

When in doubt → `regression/`. A test with a clear ticket number ages well; scenarios need maintenance forever.

## The `@scenario` decorator

Every new test should use this unless you have a specific reason not to.

```python
@scenario(
    mode="local",                    # or "cloud" — cloud auto-skips without SCOPE_CLOUD_APP_ID
    workflow="local-passthrough",    # default for ctx.complete_onboarding()
    marks=(pytest.mark.chaos,),      # optional: extra pytest marks
)
def test_my_thing(ctx): ...
```

The decorator pulls five fixtures, constructs the `ctx`, and installs a teardown that:

1. **Stops the stream cleanly** (marks an initiated stop, clicks Stop — no-op if already stopped).
2. **Populates report dimensions** via `gates.enforce_all_gates` (retry_count, unexpected_close_count, ui_error_events).
3. **Asserts zero hard-fails** — retries, unexpected closes, or UI errors fail the test.

You almost never need to write `mark_initiated_stop()`, `enforce_all_gates()`, or the five-fixture signature. If you do, see "Escape hatches" below.

## `ctx` surface

The minimal API that covers ~80% of what tests need:

| Action | Call |
|---|---|
| Drive onboarding to the graph view | `ctx.complete_onboarding()` |
| Click Run, wait for first frame (records `first_frame_time_ms`) | `ctx.run_and_wait_first_frame(timeout_ms=60_000)` |
| Mark initiated stop + click Stop (idempotent) | `ctx.stop_stream()` |
| Click Run/Stop without waiting (chaos loops) | `ctx.toggle_run()` |
| Send a parameter update over HTTP (returns status code) | `ctx.set_parameter("__prompt", "hi")` |
| Read current parameter state | `ctx.get_parameters()` |
| Fetch session metrics (fps, VRAM, frame stats) | `ctx.metrics()` |
| Click / wait on a `data-testid` | `ctx.click("testid")`, `ctx.wait("testid")` |
| Deterministic browser sleep | `ctx.sleep(ms)` |
| Get a chaos driver seeded for reproducibility | `ctx.chaos()` |
| Record a report dimension | `ctx.measure("my_ms", 42)` |
| Stash metadata on the report | `ctx.metadata("workflow", "custom")` |

### Escape hatches (when `ctx` isn't enough)

Everything below is a first-class attribute on `ctx`:

| Attr | Use for |
|---|---|
| `ctx.driver` | Full `PlaywrightDriver` (tour handling, error-toast counts). |
| `ctx.page` | Raw Playwright `Page`. Locators, evaluate, assertions. |
| `ctx.harness` | `ScopeHarness`. Mostly: `ctx.harness.log_path`, `ctx.harness.tmp_dir`. |
| `ctx.base_url` | `http://127.0.0.1:<port>`. For raw `requests.post`. |
| `ctx.retry_probe` | Inspect retry counters between phases, call `reset()` after warmup. |
| `ctx.failure_watcher` | `mark_initiated_stop()` when you stop by a non-standard path. |
| `ctx.report` | Add custom hard-fails via `ctx.report.fail("reason")`. |

## Data-testid map

Keep this list aligned with the frontend as new anchors are added. Source of truth: `grep -r 'data-testid' frontend/src`.

| Testid | Where | Purpose |
|---|---|---|
| `inference-mode-local`, `inference-mode-cloud` | Onboarding step 1 | Provider selection |
| `inference-mode-continue` | Onboarding step 1 | Advance from provider selection |
| `telemetry-accept`, `telemetry-decline` | Onboarding telemetry modal | Consent choice |
| `workflow-card-<id>` | Workflow picker | One per starter workflow (`local-passthrough`, `starter-mythical-creature`, `starter-ref-image`, `starter-ltx-text-to-video`) |
| `workflow-get-started` | Workflow picker | Confirm selection |
| `workflow-import-load` | Post-pick dialog | Load the imported workflow |
| `tour-next`, `tour-skip` | In-graph tour tooltips | Advance / dismiss onboarding tour |
| `stream-run-stop` | Graph toolbar | Run/Stop toggle (`data-streaming="true"` when active) |
| `sink-video` | Sink node | The `<video>` element that renders the pipeline output |
| `cloud-toggle` | Settings | Cloud mode toggle |
| `start-stream-button` | WebRTC connect panel | Secondary start path |

Workflow IDs come from `frontend/src/components/onboarding/starterWorkflows.ts`. CPU-only PR ring currently uses only `local-passthrough`; GPU workflows run nightly.

## Fixture dependency (if you read no other diagram)

```
scope_harness  (per-test subprocess, isolated DAYDREAM_SCOPE_DIR, SCOPE_TEST_INSTRUMENTATION=1)
  ├── driver          (Playwright context → Scope URL; installs cloud auth bypass when @cloud)
  ├── retry_probe     (talks to /api/v1/_debug/retry_stats)
  ├── failure_watcher (tails scope.log for session_closed / CRITICAL)
  └── report          (emits report.json on teardown)
```

`@scenario` pulls all five. `conftest.py` also runs an autouse `enforce_contracts` hook that fails any test where a retry counter ticked or an unexpected close fired — even if the body forgot to check.

## Mode selection

- `mode="local"` — default. Fully self-contained, CPU-only pipelines work (passthrough).
- `mode="cloud"` — requires env `SCOPE_CLOUD_APP_ID=<fal-app>`. The test auto-skips otherwise. The `driver` fixture pre-seeds a localStorage auth blob via `harness.cloud_auth.install_cloud_auth_bypass` so sign-in advances automatically.

Cloud workflows available: `starter-mythical-creature`, `starter-ref-image`, `starter-ltx-text-to-video`.

## Gotchas (every one of these cost someone an afternoon)

1. **Never call `session.close()` or kill the Scope process mid-test without `failure_watcher.mark_initiated_stop()` first.** The watcher will flag the close as unexpected and your test will red with a confusing teardown error. `ctx.stop_stream()` already does this for you — prefer that.
2. **Don't call `retry_probe.reset()` inside the body for "warmup"** unless you understand that you're losing evidence. Prefer designing the test so the warmup phase shouldn't tick any counters.
3. **Cloud tests are real network calls.** A flaky fal app = red tests. Check `SCOPE_CLOUD_APP_ID` points at a healthy deployment (`curl $FAL_URL/ws`) before blaming the test.
4. **First-frame timeout varies by mode.** Local is <15s, cloud can be 30–60s on a cold load. `ctx.run_and_wait_first_frame()` defaults to 60s; bump `timeout_ms` if you see spurious timeouts.
5. **Parameter updates race with the WebRTC data channel.** If you spam parameters faster than the channel can drain, the backend still accepts them via HTTP — that's intentional. Don't assert `get_parameters()` immediately after `set_parameter()`; poll.
6. **Do not `pytest.mark.cloud(test)` manually.** Pass `mode="cloud"` to `@scenario` instead; the decorator applies the marker AND makes `ctx.complete_onboarding()` dispatch to the cloud flow.
7. **Test file names must start with `test_`.** pytest collection rule; the decorator can't override it.

## Running

```bash
# once
uv sync --group product-tests
uv run playwright install chromium

# the test you just wrote
uv run pytest product-tests/regression/test_pr_1234_parameter_spam.py -v

# everything
uv run pytest product-tests/ -v

# reproducible chaos
uv run pytest product-tests/chaos/ --chaos-seed=abc123

# cloud
SCOPE_CLOUD_APP_ID=<fal-app-id> uv run pytest product-tests/scenarios/test_onboarding_cloud.py -v
```

Reports land in `product-tests/reports/<run-id>/` with per-test `report.json`, `scope.log`, browser `video.webm`, and Playwright `trace.zip`. A run-level `summary.md` is emitted for PR comments.

## Let Claude write it for you

If you'd rather describe the bug in English than write the test: use the `/product-test-writer` skill. Give it a bug description (or a PR number); it produces a runnable file in `regression/` using these same templates and the testid map above.
