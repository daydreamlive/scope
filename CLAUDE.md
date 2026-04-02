# CLAUDE.md

## Project Overview

Daydream Scope is a tool for running real-time, interactive generative AI video pipelines. It uses a Python/FastAPI backend with a React/TypeScript frontend with support for multiple autoregressive video diffusion models with WebRTC streaming. The frontend and backend are also bundled into an Electron desktop app.

## Development Commands

### Server (Python)

```bash
uv sync --group dev          # Install all dependencies including dev
uv run pre-commit install    # Install pre-commit hooks (required)
uv run daydream-scope --reload  # Run server with hot reload (localhost:8000)
uv run pytest                # Run tests
```

For all Python related commands use `uv run python`.

### Frontend (from frontend/ directory)

```bash
npm install                  # Install dependencies
npm run dev                  # Development server with hot reload
npm run build                # Build for production
npm run lint:fix             # Fix linting issues
npm run format               # Format with Prettier
```

### Build & Test

```bash
uv run build                 # Build frontend and Python package
PIPELINE=longlive uv run daydream-scope  # Run with specific pipeline auto-loaded
uv run -m scope.core.pipelines.longlive.test  # Test specific pipeline
```

## Architecture

### Backend (`src/scope/`)

- **`server/`**: FastAPI application, WebRTC streaming, model downloading
- **`core/`**: Pipeline definitions, registry, base classes

Key files:

- **`server/app.py`**: Main FastAPI application entry point
- **`server/pipeline_manager.py`**: Manages pipeline lifecycle with lazy loading
- **`server/webrtc.py`**: WebRTC streaming implementation
- **`core/pipelines/`**: Video generation pipelines (each in its own directory)
  - `interface.py`: Abstract `Pipeline` base class - all pipelines implement `__call__()`
  - `registry.py`: Registry pattern for dynamic pipeline discovery
  - `base_schema.py`: Pydantic config base classes (`BasePipelineConfig`)
  - `artifacts.py`: Artifact definitions for model dependencies

### Frontend (`frontend/src/`)

- React 19 + TypeScript + Vite
- Radix UI components with Tailwind CSS
- Timeline editor for prompt sequencing

### Desktop (`app/`)

- **`main.ts`**: App lifecycle, IPC handlers, orchestrates services
- **`pythonProcess.ts`**: Spawns Python backend via `uv run daydream-scope --port 52178`
- **`electronApp.ts`**: Window management, loads backend's frontend URL when server is ready
- **`setup.ts`**: Downloads/installs `uv`, runs `uv sync` on first launch

Electron main process → spawns Python backend → waits for health check → loads `http://127.0.0.1:52178` in BrowserWindow. The Electron renderer initially shows setup/loading screens, then switches to the Python-served frontend once the backend is ready.

### Key Patterns

- **Pipeline Registry**: Centralized registry eliminates if/elif chains for pipeline selection
- **Lazy Loading**: Pipelines load on demand via `PipelineManager`
- **Thread Safety**: Reentrant locks protect pipeline access
- **Pydantic Configs**: Type-safe configuration using Pydantic models

### Additional Documentation

This documentation can be used to understand the architecture of the project:

- The `docs/api` directory contains server API reference
- The `docs/architecture` contains architecture documents describing different systems used within the project
- Additional agent-specific instructions and reusable skills can be found in `.agents/skills`

### Tempo Sync (Link / MIDI)

- Python extras: `uv sync --extra link` (Ableton Link) or `uv sync --extra midi` (MIDI clock).
- On Linux, the ALSA library is required: install `libasound2` (Debian/Ubuntu), `alsa-lib` (Fedora/RHEL), or `alsa-lib` (Arch). Docker images do not include ALSA since MIDI requires local hardware access.

## Local Cloud Testing

Test the cloud relay flow locally by running two Scope instances — one acting as the "cloud" relay server.

**Environment variables:**

- `SCOPE_CLOUD_WS=1` — enables the `/ws` WebSocket endpoint on a Scope instance, making it act as a cloud relay server
- `SCOPE_CLOUD_WS_URL` — overrides the cloud WebSocket URL so the connecting instance points to your local "cloud" instead of fal.ai
- `SCOPE_CLOUD_APP_ID` — any non-empty value (e.g., `local`) to satisfy the app ID requirement

**Setup (two terminals):**

```bash
# Terminal 1 — "cloud" instance (relay server):
SCOPE_CLOUD_WS=1 uv run daydream-scope --port 8002

# Terminal 2 — "local" instance (connects to cloud):
SCOPE_CLOUD_WS_URL=ws://localhost:8002/ws SCOPE_CLOUD_APP_ID=local uv run daydream-scope --port 8022
```

Open <http://localhost:8022>, connect to cloud from the UI, load a pipeline, and start streaming. The local instance connects via WebSocket to the "cloud" instance on port 8002, which proxies WebRTC signaling and API requests back to itself.

**Key files:**

- `cloud/dev_app.py` — development-only WebSocket handler mimicking the fal.ai cloud protocol
- `server/cloud_connection.py` — client-side connection manager (`SCOPE_CLOUD_WS_URL` override in `_build_ws_url()`)
- `server/mcp_router.py` — headless session endpoints and cloud output wiring (`_wire_cloud_outputs`)
- `server/cloud_webrtc_client.py` — WebRTC client that sends frames to cloud and receives output
- `server/cloud_relay.py` — frame relay between FrameProcessor and cloud WebRTC
- `server/headless.py` — HeadlessSession with frame consumer and per-sink frame capture
- `server/sink_manager.py` — per-sink queue routing and recording coordination
- `server/graph_executor.py` — graph validation and pipeline wiring
- `server/pipeline_manager.py` — pipeline loading and aliasing (node_id → pipeline_id mapping)

**Cloud frame flow architecture (local cloud dev):**

```
Local (8022)                              Cloud (8002)
─────────────                             ────────────
SourceManager reads video files
  → FrameProcessor._on_hardware_source_frame()
  → CloudRelay.send_frame_to_source()
  → CloudWebRTCClient.input_tracks[i]    → WebRTC track received
     (WebRTC)                             → VideoProcessingTrack.recv()
                                          → FrameProcessor.put_to_source()
                                          → GraphExecutor processes pipeline(s)
                                          → SinkOutputTrack(s) send output
  CloudWebRTCClient receives tracks ←     ← WebRTC output tracks
  output_handlers[0] = primary sink       (track 0: primary sink)
  output_handlers[1..N] = extra sinks     (track 1+: extra sinks, record nodes)
  _wire_cloud_outputs() routes to:
    - sink_manager._sink_queues_by_node (per-sink queues)
    - recording_coordinator queues (per-record-node)
  HeadlessSession._consume_frames()
    reads from per-sink queues → _last_frames_by_sink
```

## MCP Server Testing

When asked to test Scope via MCP tools (e.g., with a workflow JSON), follow this sequence directly — do not read source code to figure out the API. Use the HTTP API directly (not MCP tools) because restarting Scope kills the MCP server connection.

**IMPORTANT — single instance by default:** Unless the user explicitly asks for "local cloud", "cloud testing", or "two instances", run a SINGLE Scope instance. Do NOT set up the local cloud relay (two-instance) architecture unless explicitly requested.

**Setup (single instance):**

```bash
lsof -ti:8022 | xargs kill -9 2>/dev/null
CUDA_VISIBLE_DEVICES="" uv run daydream-scope --port 8022 > /tmp/scope.log 2>&1 &
# Wait for healthy:
for i in $(seq 1 30); do curl -s http://localhost:8022/health > /dev/null 2>&1 && break; sleep 1; done
```

**Test sequence (HTTP API):**

1. Resolve workflow: `POST /api/v1/workflow/resolve` body: `{"pipelines": [...]}` (pipelines array from workflow JSON, NOT wrapped)
2. Load pipelines: `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["split-screen", "passthrough"]}` (array of unique IDs)
3. Wait for load: `GET /api/v1/pipeline/status` — poll until `"status": "loaded"`
4. Start session: `POST /api/v1/session/start` (see below for body format)
5. Wait ~10s for frames to flow, verify with `GET /api/v1/session/metrics`
6. Capture frames: `GET /api/v1/session/frame?sink_node_id=<id>` — save to `/tmp/frame_<sink_id>.jpg`
7. Start per-node recording: `POST /api/v1/recordings/headless/start?node_id=<record_id>`
8. Wait ~5s
9. Stop recording: `POST /api/v1/recordings/headless/stop?node_id=<record_id>`
10. Download recording: `GET /api/v1/recordings/headless?node_id=<record_id>` — save to `/tmp/recording_<record_id>.mp4`
11. Stop session: `POST /api/v1/session/stop`

**Critical: `input_mode: "video"` is required for video file sources.** Without it, `CloudRelay.video_mode` stays False and no frames are sent. Always include `"input_mode": "video"` in the session start body when using video file inputs.

**Session start body for multi-source/multi-sink graphs:**

```json
{
  "input_mode": "video",
  "graph": {
    "nodes": [
      {"id": "input", "type": "source", "source_mode": "video_file", "source_name": "/tmp/test.mp4"},
      {"id": "my_pipeline", "type": "pipeline", "pipeline_id": "split-screen"},
      {"id": "output", "type": "sink"},
      {"id": "record", "type": "record"}
    ],
    "edges": [
      {"from": "input", "from_port": "video", "to_node": "my_pipeline", "to_port": "video", "kind": "stream"},
      {"from": "my_pipeline", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"},
      {"from": "my_pipeline", "from_port": "video", "to_node": "record", "to_port": "video", "kind": "stream"}
    ]
  }
}
```

Record nodes and their edges come from the workflow JSON's `graph.ui_state.nodes` (type=record) and `graph.ui_state.edges`. Convert UI edge format to API edge format: `source` → `from`, `sourceHandle: "stream:video"` → `from_port: "video"`, `target` → `to_node`, `targetHandle: "stream:video"` → `to_port: "video"`, add `kind: "stream"`.

**Session start body for single pipeline:**

```json
{
  "pipeline_id": "passthrough",
  "input_mode": "video",
  "input_source": {
    "enabled": true,
    "source_type": "video_file",
    "source_name": "/tmp/test_input.mp4"
  }
}
```

**Create test videos** (ffmpeg not available, use OpenCV):

```bash
uv run python -c "
import cv2, numpy as np
for name, color in [('test', (0,0,255)), ('test1', (0,255,0)), ('test2', (255,0,0))]:
    w = cv2.VideoWriter(f'/tmp/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
    frame = np.zeros((512,512,3), dtype=np.uint8); frame[:] = color
    for _ in range(300): w.write(frame)
    w.release()
"
```

**Per-node recording:** The session ID for headless sessions is `"headless"`. Recording endpoints require `?node_id=` for graphs with record nodes:

| Operation | HTTP API |
|-----------|----------|
| Start recording | `POST /api/v1/recordings/headless/start?node_id=record` |
| Stop recording | `POST /api/v1/recordings/headless/stop?node_id=record` |
| Download recording | `GET /api/v1/recordings/headless?node_id=record` (returns MP4 binary) |

**Full HTTP API reference:**

| Operation | HTTP API |
|-----------|----------|
| Health | `GET /health` (NOT `/api/v1/health`) |
| Resolve workflow | `POST /api/v1/workflow/resolve` body: `{"pipelines": [...]}` |
| Load pipeline | `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["name"]}` |
| Pipeline status | `GET /api/v1/pipeline/status` |
| Start session | `POST /api/v1/session/start` body: `{"input_mode": "video", "graph": {...}}` |
| Session metrics | `GET /api/v1/session/metrics` |
| Capture frame | `GET /api/v1/session/frame` or `?sink_node_id=output` (returns JPEG binary) |
| Stream MPEG-TS | `GET /api/v1/session/output.ts` (streams `video/mp2t`; includes audio when pipeline produces it) |
| Stop session | `POST /api/v1/session/stop` |
| Start recording | `POST /api/v1/recordings/headless/start?node_id=<id>` |
| Stop recording | `POST /api/v1/recordings/headless/stop?node_id=<id>` |
| Download recording | `GET /api/v1/recordings/headless?node_id=<id>` (returns MP4 binary) |
| Logs | `GET /api/v1/logs/tail?lines=30` |

**Debugging:**

- Check frame flow with `GET /api/v1/session/metrics`
- Check logs with `GET /api/v1/logs/tail?lines=50`
- `frames_in > 0, frames_out = 0` → pipeline processing failing
- All sinks return same frame → per-sink routing issue in HeadlessSession

## MCP Server Testing with Local Cloud Dev

**Only use this section when the user explicitly asks for local cloud / two-instance testing.**

Test the cloud relay flow locally by running two Scope instances — one acting as the "cloud" relay server. This is for testing the cloud WebRTC relay path specifically.

**Setup (two instances):**

```bash
lsof -ti:8002 -ti:8022 | xargs kill -9 2>/dev/null

# Cloud instance (start first):
CUDA_VISIBLE_DEVICES="" SCOPE_CLOUD_WS=1 uv run daydream-scope --port 8002 > /tmp/cloud.log 2>&1 &
for i in $(seq 1 30); do curl -s http://localhost:8002/health > /dev/null 2>&1 && break; sleep 1; done

# Local instance (start after cloud is healthy):
CUDA_VISIBLE_DEVICES="" SCOPE_CLOUD_WS_URL=ws://localhost:8002/ws SCOPE_CLOUD_APP_ID=local uv run daydream-scope --port 8022 > /tmp/local.log 2>&1 &
for i in $(seq 1 30); do curl -s http://localhost:8022/health > /dev/null 2>&1 && break; sleep 1; done
```

**Additional cloud-specific steps (before resolve/load):**

```bash
# Connect to cloud:
curl -s -X POST http://localhost:8022/api/v1/cloud/connect -H 'Content-Type: application/json' -d '{"app_id": "local"}'
# Wait and verify:
sleep 2 && curl -s http://localhost:8022/api/v1/cloud/status
```

Then follow the same test sequence as single-instance mode above. All session/frame/recording endpoints go to port 8022 (local), not 8002 (cloud). Pipeline load is automatically proxied to cloud.

**Cloud-specific debugging:**

- `frames_to_cloud > 0, frames_from_cloud = 0` → cloud is not sending output back; check cloud logs
- Both instances write separate log files to `~/.daydream-scope/logs/` — the `/api/v1/logs/tail` endpoint returns the most recent file alphabetically, which may be the wrong instance's logs. Read the actual log files with `ls -t ~/.daydream-scope/logs/scope-logs-*.log | head -2` to find both
- Cloud status: `GET /api/v1/cloud/status` on port 8022

## Contributing Requirements

- All commits must be signed off (DCO): `git commit -s`
- Pre-commit hooks run ruff (Python) and prettier/eslint (frontend)
- Models stored in `~/.daydream-scope/models` (configurable via `DAYDREAM_SCOPE_MODELS_DIR`)

## Style Guidelines

### Backend

- Use relative imports if it is single or double dot (eg .package or ..package) and otherwise use an absolute import
- `scope.server` can import from `scope.core`, but `scope.core` must never import from `scope.server`

## Verifying Work

Follow these guidelines for verifying work when implementation for a task is complete. **Always run lint checks before committing, pushing, or finalizing any changes.**

### Backend

- Run `uv run ruff check src/` to lint Python code. Use `uv run ruff check --fix src/` to auto-fix issues.
- Run `uv run ruff format --check src/` to verify formatting. Use `uv run ruff format src/` to auto-fix.
- Use `uv run daydream-scope` to confirm that the server starts up without errors.

### Frontend

- Run `npm run lint` (from `frontend/`) to check for lint errors. Use `npm run lint:fix` to auto-fix.
- Run `npm run format:check` (from `frontend/`) to verify formatting. Use `npm run format` to auto-fix.
- Use `npm run build` (from `frontend/`) to confirm that builds work properly.
