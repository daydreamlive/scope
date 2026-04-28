---
name: testing-livepeer-fal-deploy
description: End-to-end test harness for Scope's Livepeer cloud path against a deployed fal.ai app — the only supported cloud path going forward (the old cloud-relay / direct mode using `fal_app.py` + `CloudConnectionManager` is being deprecated). Primary path is a Playwright browser test that drives the full UI flow (camera → local scope WebRTC → livepeer trickle → fal runner → back), producing every session-lifecycle Kafka event. Secondary path is `test-cloud-connect.sh` — a bash/curl smoke test for the `/api/v1/cloud/connect` path only. TRIGGER any time a user says "test cloud", "test the fal deploy", "test cloud streaming", "run the e2e test", "run playwright", "verify cloud connect", "verify kafka events", "diagnose fal", "debug fal deploy", "did my stream work", "deploy-staging.sh", OR pastes any of these errors — "All orchestrators failed (N tried)", "ACCESS_DENIED", "did not receive ready message from websocket", "discover_orchestrators requires discovery_url", "cold start" — OR has just changed `src/scope/cloud/livepeer_fal_app.py` / `src/scope/cloud/livepeer_app.py` / `src/scope/server/livepeer.py` / `src/scope/server/livepeer_client.py`. Use `testing-livepeer` instead for a fully-local livepeer stack (prebuilt go-livepeer binary, no fal involvement).
---

# Testing Livepeer fal Deploy

## When to use

Use when testing the **deployed** livepeer path end-to-end — local Scope
client → daydream orchestrator → deployed fal app. This exercises:

- The wrapper in `src/scope/cloud/livepeer_fal_app.py` that fal runs
- The runner in `src/scope/cloud/livepeer_app.py` that spawns inside the
  fal container
- The orchestrator → fal handshake (headers, auth, cold start)
- Kafka event publishing across wrapper + runner (full lifecycle)

**Two paths, pick the right one:**

- **Playwright (primary)** — real browser drives the Perform-mode UI
  with a synthetic camera, streams through, verifies the output video
  comes back from the cloud. This is the only path that exercises the
  full livepeer trickle round-trip and produces every lifecycle Kafka
  event (`pipeline_loaded`, `session_created`, `stream_started`,
  `stream_heartbeat`, `session_closed`). Takes 2–5 minutes.
- **`test-cloud-connect.sh` (secondary, HTTP-only)** — bash script that
  POSTs `/api/v1/cloud/connect` and polls `/api/v1/cloud/status`. Only
  verifies the `websocket_connected` / `websocket_disconnected` pair at
  the wrapper layer. Useful as a fast smoke test ("did the container
  come up?") or in `git bisect run` against cloud-connect regressions.
  Does not produce pipeline/session/stream events.

Do **not** use this skill for local-only livepeer testing — that's
`testing-livepeer` (prebuilt go-livepeer + local runner, no fal).

## One-time setup

1. **`.env.local`**: copy `.env.example` to `.env.local` (gitignored)
   and fill in real values:
   - `SCOPE_CLOUD_APP_ID` — your fal app URL. For the default `main`
     env, the URL does **not** include a `--main` suffix (e.g.
     `daydream/scope-livepeer-emran/ws`). Non-default envs do include
     the suffix (e.g. `--preview/ws`).
   - `SCOPE_CLOUD_API_KEY` — daydream cloud API key (sk_...). Without
     this the scope client can't hit `signer.daydream.live` and fails
     with `discover_orchestrators requires discovery_url or signer_url`.
   - `SCOPE_USER_ID` — daydream user id. The runner's
     `validate_user_access` rejects with `ACCESS_DENIED` when missing.
     Find it in `~/.daydream-scope/logs/scope-logs-*.log` after a
     successful UI connect, or in devtools Network on
     `/api/v1/cloud/connect`.
   - (Optional) `LIVEPEER_DEBUG=1` — surfaces per-orchestrator
     rejection reasons in scope.log; essential for diagnosing
     `All orchestrators failed (N tried)`.
2. **product-tests setup** (once per machine):
   ```bash
   uv sync --group product-tests
   uv run playwright install --with-deps chromium
   ```
   This installs pytest, Playwright, and Chromium with the right
   system deps. Without `--with-deps`, the browser fails to launch
   with `error while loading shared libraries: libnspr4.so`.

   The `@scenario(mode="cloud")` decorator on the test handles auth
   via a localStorage bypass (see ``harness/cloud_auth.py``), so no
   frontend rebuild with `VITE_DAYDREAM_API_KEY` is needed.

## Running the Playwright test (primary)

When the user says "test cloud" (or any trigger in the description),
**always deploy their current working tree before running Playwright**.
Otherwise the test runs against whatever stale code was last deployed
and can false-positive on their change.

### Step 0 — Ask the user where to deploy

Before anything else, confirm the deploy target. Use AskUserQuestion
(or plain text prompts) and persist answers for the session:

1. **Fal app name** — required. If `SCOPE_FAL_APP_NAME` is set in
   `.env.local`, show that value and ask the user to confirm or
   override. Otherwise ask outright (e.g. `scope-livepeer-<name>`).
2. **Fal env** — defaults to `main`. If `SCOPE_FAL_ENV` is set in
   `.env.local`, show and offer to override. Non-default envs (e.g.
   `preview`) change the URL suffix in `SCOPE_CLOUD_APP_ID` — see
   below.

Once confirmed, export both for the current shell, and derive /
overwrite `SCOPE_CLOUD_APP_ID`:

| Env | `SCOPE_CLOUD_APP_ID` |
|---|---|
| `main` | `daydream/<app>/ws`         (no suffix) |
| anything else | `daydream/<app>--<env>/ws`  (with suffix) |

This is a fal convention — the default `main` env is exposed without
a suffix; all other envs include `--<env>` in the URL. Getting this
wrong produces `did not receive ready message from websocket`.

### Step 1 — Sanity-check `.env.local`

- `SCOPE_CLOUD_API_KEY` must be set (otherwise:
  `discover_orchestrators requires discovery_url or signer_url`)
- `SCOPE_USER_ID` must be set (otherwise the runner's
  `validate_user_access` rejects with `ACCESS_DENIED`)

If either is missing, stop and ask the user before deploying.

### Step 2 — Kill any scope already on :8000

If another scope process is bound to the port, stop it (or ask the
user) before continuing. The run-app.sh the script starts must be the
one under test.

### Step 3 — Deploy

```bash
SCOPE_FAL_APP_NAME=<app> SCOPE_FAL_ENV=<env> ./deploy-staging.sh
```

Abort with a clear error if this fails — don't run Playwright against
stale deployed code. Common failure: the `{git-short-sha}-cloud`
Docker base image isn't built yet (CI for the current commit is still
running). If that's the case, either wait for CI or have the user
confirm they want to deploy against an older base image.

### Step 4 — Run the cloud-streaming test

The test spins up its own fresh scope subprocess per test (via the
``scope_harness`` fixture), so you don't run ``./run-app.sh`` first —
just point ``SCOPE_CLOUD_APP_ID`` at the deploy and let pytest do it.

```bash
SCOPE_CLOUD_APP_ID=<derived-url> \
  uv run pytest product-tests/release/test_cloud_streaming.py -v -m cloud
```

Reports land in ``product-tests/reports/<run-id>/`` (per-test
``report.json``, ``trace.zip``, video, ``scope.log``, plus a
top-level ``summary.md``).

Expected on success (≤5 min cold, ~20 s warm):

```
product-tests/release/test_cloud_streaming.py::test_cloud_streaming_perform_mode_passthrough PASSED
============ 1 passed in <duration> ============
```

The summary.md at ``product-tests/reports/<run-id>/summary.md``
records ``retry_count``, ``unexpected_close_count``, and
``ui_error_events`` — all should be zero for a clean run.

**What the test does in livepeer terms:**

1. Navigates to `localhost:8000`, switches the UI to Perform mode.
2. Opens settings, flips Remote Inference on, waits for Connection ID
   (proves the fal WebSocket handshake completed and
   `websocket_connected` fired in Kafka).
3. Selects the `passthrough` pipeline — triggers `pipeline/load`, which
   runs on the fal runner and emits `pipeline_load_start` +
   `pipeline_loaded`.
4. Switches the input source to Camera — Playwright's launch args
   `--use-fake-device-for-media-stream` and
   `--use-fake-ui-for-media-stream` (configured in the ``driver``
   fixture in ``product-tests/conftest.py``) give ``getUserMedia()``
   a synthetic feed.
   This is essential: without a real MediaStream, the browser↔local
   scope WebRTC ICE never completes, `CloudTrack._start()` is never
   called, and the runner never gets `start_stream`.
5. Clicks the play overlay (`[data-testid="start-stream-button"]`).
   Frames flow via livepeer trickle through the orchestrator to the
   fal runner; the runner emits `session_created` and `stream_started`.
6. Waits 15 s so at least one `stream_heartbeat` fires on the runner.
7. Asserts the **output** `<video>` inside the "Video Output" card is
   actively playing (`currentTime > 0`). Checking any `<video>` would
   false-positive on the local input preview.
8. Stops the stream. Runner emits `session_closed` and eventually
   `websocket_disconnected` when the session is reaped.

## Running the quick HTTP smoke (secondary)

```bash
./test-cloud-connect.sh [flags]
```

Flags: `--skip-push`, `--skip-build-wait`, `--skip-deploy`,
`--keep-scope`, `--port N`. Env overrides:
`TIMEOUT_CONNECT`, `TIMEOUT_HEALTH`, `TIMEOUT_CI`, etc.

Exit codes (bisect-friendly — `git bisect run` works):

| Code | Meaning |
|---|---|
| 0 | Connected to cloud |
| 1 | Cloud reported an `error` in `/cloud/status` |
| 2 | Timed out waiting for connect |
| 3 | Infra failure (push / CI / deploy / scope startup) |

This only hits `POST /api/v1/cloud/connect` and polls status — it does
**not** start a stream, load a pipeline on the cloud, or produce the
session/stream events. If those are what you're after, use Playwright.

A `--full-session` flag exists but hits a known gap: `/api/v1/session/start`
is not livepeer-compatible (TODO at `src/scope/server/mcp_router.py:252`)
and will error with `Pipeline X not loaded` in livepeer mode. The
Playwright path is the supported way to exercise a full session.

## Logs

- `/tmp/test-cloud-connect/scope.log` — local scope stdout/stderr
  (grep for `livepeer_gateway` when `LIVEPEER_DEBUG=1`)
- `~/.daydream-scope/logs/scope-logs-*.log` — scope's rolling app logs
- `product-tests/reports/<run-id>/` — per-test ``report.json``,
  Playwright video, ``trace.zip``, and the scope subprocess
  ``scope.log``. Plus a top-level ``summary.md``.
- fal dashboard — runner stdout/stderr, including `[Kafka] Published
  event: …` lines from `scope.server.kafka_publisher` in the runner.
  Not accessible via CLI; open <https://fal.ai/dashboard/logs>.

## Common failure signatures

- **`All orchestrators failed (N tried)`** — set `LIVEPEER_DEBUG=1` to
  get the per-orchestrator reason. Typical root causes:
  - `did not receive ready message from websocket` → fal URL wrong
    (e.g. stray `--main` suffix) or container cold-starting.
  - `serverless handshake failed (ACCESS_DENIED)` → runner's
    `validate_user_access` rejected (missing `SCOPE_USER_ID`, or
    daydream API couldn't find the user).
- **`discover_orchestrators requires discovery_url or signer_url`** →
  `SCOPE_CLOUD_API_KEY` not set; signer fallback isn't configured.
- **Playwright: `error while loading shared libraries: libnspr4.so`** →
  Chromium system deps missing; re-run
  `uv run playwright install --with-deps chromium` from setup.
- **Playwright: test passes but ClickHouse only has
  `websocket_connected`** — the test probably clicked stop before ICE
  completed. Confirm the fake-device launch args are set and the
  Camera input was selected (not File).
- **Playwright: `FrameProcessor failed to start: Pipeline X not
  loaded`** — you're running the HTTP script's `--full-session` flag,
  not the cloud-streaming test. Switch to
  `uv run pytest product-tests/release/test_cloud_streaming.py -v -m cloud`.

## What "round-trip verified" looks like in ClickHouse

After a successful Playwright run, `scope_cloud_events` filtered by
your `user_id` and the `connection_id` from the `websocket_connected`
row should contain:

```
websocket_connected          (wrapper)
pipeline_load_start          (runner)
pipeline_loaded              (runner)
session_created              (runner)
stream_started               (runner)
stream_heartbeat × 1..N      (runner, ~every 10 s)
stream_stopped               (runner)
session_closed               (runner)
websocket_disconnected       (wrapper, on session reap)
```

All sharing the same `user_id` and `connection_id` (= `manifest_id`).
If any runner-emitted row is missing, something in
`src/scope/cloud/livepeer_app.py` regressed — check the FrameProcessor
construction around the `start_stream` handler and the explicit
`publish_event` calls for `session_created` / `session_closed`.
