---
name: testing-livepeer
description: Test Scope in Livepeer mode end to end using a prebuilt `go-livepeer` artifact from the `ja/serverless` PR, `uv run --extra livepeer livepeer-runner`, and Scope with `SCOPE_CLOUD_MODE=livepeer`. Use when the user mentions Livepeer mode, `livepeer-runner`, `SCOPE_CLOUD_MODE=livepeer`, or local serverless testing. Pair with `testing-scope-mcp` when the test path should be driven through Scope's MCP server.
---

# Testing Livepeer

## Quick Start

Use this skill to verify or debug Scope against a fully local Livepeer stack — orchestrator, runner, and Scope all run on the same machine. This is not for testing against remote or production Livepeer infrastructure. For MCP-driven testing, spin up the stack below then follow the `testing-scope-mcp` skill.

Default assumptions:
- Use a prebuilt `go-livepeer` binary from the `ja/serverless` PR artifacts — never build from source
- Run the Livepeer orchestrator locally on `localhost:8935`
- Run `livepeer-runner` with `LIVEPEER_DEV_MODE=1 SCOPE_PORT=9001 UV_NO_SYNC=1 uv run --extra livepeer livepeer-runner`
- Launch Scope with `SCOPE_CLOUD_MODE=livepeer`
- Keep iterating until the end-to-end path under test actually works

## Standard Workflow

1. Download a prebuilt `go-livepeer` artifact for this machine.
2. Create `serverless.json` with the runner websocket config.
3. Start `go-livepeer`, then `livepeer-runner`, then Scope in Livepeer mode.
4. If testing via MCP, follow the `testing-scope-mcp` skill (connect to Scope on port `8022`).
5. Exercise the relevant Livepeer-mode behavior end to end.
6. If anything fails, inspect logs from the failing process, make the smallest fix, and rerun from the earliest affected step.

## Artifact Discovery

Source: prebuilt artifacts from the `Upload artifacts to google bucket` step in the `Build binaries` workflow for [PR #3884](https://github.com/livepeer/go-livepeer/pull/3884). This PR (`ja/serverless`) is still WIP — once it merges, update this skill to pull from a release or `master` build instead.

**Always resolve the latest head SHA dynamically** — do not hardcode a SHA. Run:

```bash
SHA=$(gh pr view 3884 --repo livepeer/go-livepeer --json headRefOid -q '.headRefOid')
echo "Latest SHA: $SHA"
```

Then construct the download URL for the current platform:

```bash
# Detect platform
ARCH=$(uname -m)
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$ARCH" in
  x86_64)  ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac
URL="https://build.livepeer.live/go-livepeer/${SHA}/livepeer-${OS}-${ARCH}.tar.gz"
echo "Download URL: $URL"
```

Full platform matrix (all use `https://build.livepeer.live/go-livepeer/${SHA}/` as base):
- `livepeer-darwin-amd64.tar.gz`
- `livepeer-darwin-arm64.tar.gz`
- `livepeer-linux-amd64.tar.gz`
- `livepeer-linux-arm64.tar.gz`
- `livepeer-linux-gpu-amd64.tar.gz`
- `livepeer-linux-gpu-arm64.tar.gz`
- `livepeer-windows-amd64.zip`

If the artifact is missing, the PR head may have changed or CI hasn't finished. Verify with `gh pr view 3884 --repo livepeer/go-livepeer --json headRefOid,statusCheckRollup`. Never build `go-livepeer` from source.

## `serverless.json`

```json
[
  {
    "pipeline": "live-video-to-video",
    "model_id": "scope",
    "url": "ws://localhost:8001/ws",
    "warm": false,
    "capacity": 10
  }
]
```

The `url` points to the local `livepeer-runner` websocket endpoint, not the Scope server.

## Start The Local Stack

Kill old processes on the relevant ports first:

```bash
lsof -ti:8001 -ti:8022 -ti:8935 | xargs kill -9 2>/dev/null
lsof -tiUDP:9001 | xargs kill -9 2>/dev/null
```

**Orchestrator:**

```bash
./livepeer -orchestrator -aiWorker -aiServerless -aiModels serverless.json -serviceAddr localhost:8935
```

**Runner** (`SCOPE_PORT=9001` prevents OSC port collision with the main Scope server; `LIVEPEER_DEV_MODE=1` skips auth for local testing):

```bash
# CRITICAL: The runner needs the websockets library for its /ws endpoint.
# Without it, the orchestrator gets "websocket: bad handshake" when connecting.
uv pip install 'uvicorn[standard]'

LIVEPEER_DEV_MODE=1 SCOPE_PORT=9001 UV_NO_SYNC=1 uv run --extra livepeer livepeer-runner
```

The runner has no `/health` endpoint; verify by checking logs and that port `8001` is listening.

**Scope** (`LIVEPEER_SIGNER=off` is intentional for local dev; do not add signing unless the user asks):

```bash
LIVEPEER_DEV_MODE=1 \
LIVEPEER_SIGNER=off \
LIVEPEER_ORCH_URL=localhost:8935 \
SCOPE_CLOUD_MODE=livepeer \
UV_NO_SYNC=1 \
uv run daydream-scope --no-browser --port 8022
```

Wait for health: `curl -s http://localhost:8022/health`

For MCP-driven testing after Scope is healthy, follow the `testing-scope-mcp` skill to start the MCP server and connect to port `8022`.

## Verification

Verify in this order, and do not skip ahead:
1. `go-livepeer` runs and accepts `serverless.json`
2. `livepeer-runner` starts on `ws://localhost:8001/ws` and stays running
3. Scope starts without import, config, or OSC port-collision errors
4. Scope operates in `SCOPE_CLOUD_MODE=livepeer`
5. The specific user-facing behavior works end to end

Do not declare success after partial recovery. The final check must cover the full Livepeer-mode scenario the user cares about. A rerun after any fix must behave consistently.

Interpretation note: a successful `connect_to_cloud` + `pipeline/load` proves the control path. A headless `start_stream` + `capture_frame` may exercise local media paths only. If the user cares about true remote media/WebRTC behavior, finish with a browser or UI-driven verification.

## Headless Session in Livepeer Mode

The headless session (`/api/v1/session/start`) runs locally on the Scope instance — it is **not** proxied to cloud. However, when Scope is connected to cloud, `POST /api/v1/pipeline/load` **is** proxied to the runner. This means the pipeline gets loaded only on the runner side, and the local headless session fails with "Pipeline passthrough not loaded".

**Workaround — load the pipeline locally before starting the session:**

1. `POST /api/v1/cloud/disconnect` — disconnect from cloud
2. `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["passthrough"]}` — loads locally (no proxy)
3. `POST /api/v1/cloud/connect` body: `{}` — reconnect to cloud
4. Wait for cloud status to show `connected: true`
5. `POST /api/v1/session/start` body: `{"pipeline_id": "passthrough", ...}` — now works

This applies to all pipelines in Livepeer mode, not just passthrough.

**Video input for passthrough:** Use the bundled test video at `frontend/public/assets/test.mp4` as `input_source` with `source_type: "video_file"` and `input_mode: "video"`. Do not create a synthetic test video with ffmpeg.

## Troubleshooting

When something fails:
1. Identify the failing boundary: orchestrator / runner / Scope startup / cloud connect / websocket handshake / stream behavior.
2. Read logs from the failing process first.
3. Make the narrowest fix and restart only the affected processes.
4. Rerun the failing step, then the full end-to-end flow.

HTTP API fallbacks useful during Livepeer testing (when MCP is unavailable):

| Operation | Endpoint |
|-----------|----------|
| Cloud connect | `POST /api/v1/cloud/connect` body: `{}` |
| Cloud disconnect | `POST /api/v1/cloud/disconnect` |
| Cloud status | `GET /api/v1/cloud/status` |
| Load pipeline | `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["name"]}` |
| Pipeline status | `GET /api/v1/pipeline/status` |
| Start session | `POST /api/v1/session/start` body: `{"pipeline_id": "name", ...}` |
| Capture frame | `GET /api/v1/session/frame` (returns JPEG binary) |
| Stop session | `POST /api/v1/session/stop` |
| Start recording | `POST /api/v1/recordings/headless/start` |
| Stop recording | `POST /api/v1/recordings/headless/stop` |
| Download recording | `GET /api/v1/recordings/headless` (returns MP4 binary) |
| Logs | `GET /api/v1/logs/tail?lines=30` |

## Practical Notes

- If `uv run --extra livepeer ...` fails resolving a platform-inapplicable wheel (e.g., `flash-attn` on macOS), retry with `UV_NO_SYNC=1 uv run --extra livepeer ...`.
- If Scope cannot open `~/.daydream-scope/logs` in a sandboxed environment, set `DAYDREAM_SCOPE_LOGS_DIR` to a writable directory.
- If using the Docker artifact path instead of the native binary, ensure the container can reach the host runner on port `8001`.

## Updating This `SKILL.md`

Artifact URLs are resolved dynamically from the PR head SHA (see "Artifact Discovery"), so this file does not need manual SHA updates. If the PR merges, update the artifact discovery section to pull from a `master` build or release instead of the PR.
