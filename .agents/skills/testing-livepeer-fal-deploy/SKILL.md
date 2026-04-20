---
name: testing-livepeer-fal-deploy
description: End-to-end test harness for Scope running in Livepeer cloud mode against a deployed fal.ai app. Orchestrates git push, GitHub Actions CI build-cloud wait, deploy-staging, local scope startup, and /cloud/connect verification. Use when the user wants to test a change to `src/scope/cloud/livepeer_fal_app.py` or diagnose cloud-connect failures ("All orchestrators failed", "ACCESS_DENIED", etc.) through the real fal + orchestrator path. For a fully-local livepeer stack (no fal), use `testing-livepeer` instead.
---

# Testing Livepeer fal Deploy

## When to use

Use when testing the **deployed** livepeer cloud path end-to-end — i.e. local
Scope client → daydream orchestrator → deployed fal app. This exercises:

- The wrapper in `src/scope/cloud/livepeer_fal_app.py` that fal actually runs.
- The orchestrator → fal handshake (headers, auth, cold start).
- Kafka event publishing from the fal wrapper.

Do **not** use for local-only livepeer testing — that's what `testing-livepeer`
is for (uses a prebuilt go-livepeer binary + local runner, no fal involvement).

## One-time setup

1. Copy `.env.example` to `.env.local` and fill in real values.
   `.env.local` is gitignored — never commit it.
2. Required values:
   - `SCOPE_CLOUD_APP_ID` — your fal app. The `main` env is exposed **without**
     a `--main` suffix in the URL; non-default envs (e.g. `--preview`) include
     the suffix.
   - `SCOPE_CLOUD_API_KEY` — daydream cloud API key (sk_...). Without it the
     scope client cannot call `signer.daydream.live` and orchestrator
     discovery fails with `discover_orchestrators requires discovery_url or
     signer_url`.
   - `SCOPE_USER_ID` — daydream user id. Without it the runner's
     `validate_user_access` rejects with `ACCESS_DENIED`. Find it in
     `~/.daydream-scope/logs/scope-logs-*.log` after a successful UI connect,
     or in a browser devtools Network tab on `/api/v1/cloud/connect`.
3. Optional: `LIVEPEER_DEBUG=1` to surface per-orchestrator rejection reasons
   in the scope log (crucial for diagnosing `All orchestrators failed (N tried)`).

## Running the test

### One-shot

```bash
./test-cloud-connect.sh [flags]
```

Default flow: git push current branch → wait for GitHub Actions
`docker-build.yml build-cloud` to succeed → run `./deploy-staging.sh` →
start scope via `./run-app.sh` → POST `/api/v1/cloud/connect` → poll
`/api/v1/cloud/status` until connected, errored, or timed out.

### Flags

- `--skip-push` — don't `git push` (useful when re-testing without code
  changes, or testing `main`).
- `--skip-build-wait` — don't wait for CI (assumes the `-cloud` image is
  already built for HEAD).
- `--skip-deploy` — don't run `deploy-staging.sh` (fast iteration when only
  the scope client changed).
- `--full-session` — after connect, load a pipeline, start a session, verify
  frames flow, stop, and cloud-disconnect. **Known limitation:** in livepeer
  mode the `/api/v1/session/start` endpoint is not livepeer-compatible
  (see `TODO` at `src/scope/server/mcp_router.py:252`), so this flag hits a
  "Pipeline X not loaded" error. Use a manual UI test if you need the
  `pipeline_loaded` / `session_created` / `stream_started` / `stream_heartbeat`
  Kafka events.
- `--keep-scope` — leave scope running after the test (don't kill).
- `--port N` — change the local scope port (default 8000).

### Env overrides

`PORT`, `TIMEOUT_CONNECT` (default 180s, bump to 300+ for cold starts),
`TIMEOUT_HEALTH`, `TIMEOUT_CI`, `TIMEOUT_PIPELINE`, `TIMEOUT_FRAMES`,
`PIPELINE_ID`, `TEST_VIDEO`.

## Exit codes

Bisect-friendly — `git bisect run ./test-cloud-connect.sh` works.

| Code | Meaning |
|---|---|
| 0 | Connected (and if `--full-session`, frames flowed) |
| 1 | Cloud reported an error — see `error` field in status response |
| 2 | Timed out waiting for connect / pipeline / frames |
| 3 | Infra failure — push / CI / deploy / scope startup |
| 4 | Session-level failure (pipeline load, session start, no frames) |

## Logs

- `/tmp/test-cloud-connect/driver.log` — script's own progress log
- `/tmp/test-cloud-connect/scope.log` — stdout/stderr of the local scope
  process. If cloud connect fails, grep here for `livepeer_gateway` —
  that's where rejection reasons land when `LIVEPEER_DEBUG=1`.
- `~/.daydream-scope/logs/scope-logs-*.log` — scope's rolling app logs
  (separate from the test-captured log; useful for historical runs).
- fal deployment dashboard — the fal container's stdout/stderr, including
  `[KAFKA-DEBUG]` lines, runner subprocess logs, and Kafka publisher state.
  Not accessible via CLI; open the fal.ai dashboard for the app.

## Common failure signatures

- **`All orchestrators failed (N tried)`** — generic; enable `LIVEPEER_DEBUG=1`
  and re-run to get the specific per-orchestrator reason. Typical underlying
  causes:
  - `did not receive ready message from websocket` → fal URL wrong (e.g.
    extra `--main` suffix) or container still cold-starting.
  - `serverless handshake failed (ACCESS_DENIED)` → runner's
    `validate_user_access` rejected because `SCOPE_USER_ID` was missing
    or the daydream API couldn't find the user.
- **`discover_orchestrators requires discovery_url or signer_url`** →
  `SCOPE_CLOUD_API_KEY` not set, so the signer fallback isn't configured.
- **`FrameProcessor failed to start: Pipeline <id> not loaded`** — you used
  `--full-session` in livepeer mode. Known gap; use UI for pipeline/session
  events.

## Typical workflows

**Verify a new commit works:**
```bash
./test-cloud-connect.sh   # full cycle
```

**Fast iteration (no redeploy needed):**
```bash
./test-cloud-connect.sh --skip-push --skip-build-wait --skip-deploy
```

**Bisect a regression:**
```bash
git bisect start HEAD known-good-sha
git bisect run ./test-cloud-connect.sh --skip-push
```
(Each iteration triggers a full CI+deploy, so bisects are slow — budget
~10+ min per step.)

## What this skill does NOT cover

- Full-session event coverage (`pipeline_loaded` / `session_created` /
  `stream_started` / `stream_heartbeat`) — those fire from the cloud runner
  (or via WebRTC on the local side) and require a real streaming session
  that `/api/v1/session/start` doesn't support in livepeer mode yet. Today,
  trigger them by running scope with the UI and pushing frames through.
- ClickHouse verification — this skill produces Kafka events; use the
  user's ClickHouse MCP or dashboard to verify they land downstream.
