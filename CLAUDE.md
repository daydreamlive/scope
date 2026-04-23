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

### Cloud Modes

Scope supports two cloud backend modes controlled by `SCOPE_CLOUD_MODE`:

- **Cloud relay** (default) — `CloudConnectionManager` connects via WebSocket + WebRTC to fal.ai
- **Livepeer** (`SCOPE_CLOUD_MODE=livepeer`) — `LivepeerConnection` creates an LV2V job and streams frames over trickle HTTP channels through the Livepeer network

Both implement the same interface behind `ScopeCloudBackend` (union type in `server/scope_cloud_types.py`), so `FrameProcessor`, `CloudTrack`, and the WebRTC handler are mode-agnostic. The active backend is selected via `get_scope_cloud()` in `app.py`.

Key Livepeer files:
- **`server/livepeer.py`**: `LivepeerConnection` — high-level connection manager
- **`server/livepeer_client.py`**: `LivepeerClient` — transport layer managing LV2V job, trickle media/control channels
- **`server/cloud_track.py`**: `CloudTrack` — bridges browser WebRTC and cloud backend
- **`server/frame_processor.py`**: Routes frames to/from cloud (cloud mode detection, callbacks)

Documentation:
- Setup: `docs/livepeer.md`
- Runner/protocol architecture: `docs/architecture/livepeer.md`
- Client-side architecture: `docs/architecture/livepeer-client.md`

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

For local Livepeer cloud testing, follow `.agents/skills/testing-livepeer/SKILL.md`.

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
| List input source types | `GET /api/v1/input-sources` |
| Discover NDI sources | `GET /api/v1/input-sources/ndi/sources?timeout_ms=5000` |

**NDI sources and sinks in graphs:**

- Use `"source_mode": "ndi"` and `"source_name": "<NDI identifier>"` for NDI input sources
- Use `"sink_mode": "ndi"` and `"sink_name": "<sender name>"` for NDI output sinks
- Discover NDI sources: `GET /api/v1/input-sources/ndi/sources?timeout_ms=5000`
- NDI identifiers include a machine UUID prefix, e.g. `"69F966CD-... (MySource)"`
- Create NDI test senders with `scope.core.outputs.ndi.NDIOutputSink` (create, send_frame in loop)
- Verify NDI outputs by receiving frames with `scope.core.inputs.ndi.NDIInputSource` (connect, receive_frame)

**Example graph with NDI sources + NDI sinks:**

```json
{
  "nodes": [
    {"id": "input", "type": "source", "source_mode": "video_file", "source_name": "/tmp/test.mp4"},
    {"id": "ndi_in", "type": "source", "source_mode": "ndi", "source_name": "<NDI identifier>"},
    {"id": "my_pipeline", "type": "pipeline", "pipeline_id": "split-screen"},
    {"id": "output", "type": "sink", "sink_mode": "ndi", "sink_name": "My NDI Output"}
  ],
  "edges": [...]
}
```

**Syphon sources in graphs (macOS only):**

- Use `"source_mode": "syphon"` and `"source_name": "<display name>"` for Syphon input sources
- Discover Syphon sources: `GET /api/v1/input-sources/syphon/sources`
- Syphon display names use the format `"AppName - ServerName"` (e.g., `"TouchDesigner - Scope1"`)
- The `source_name` must match the `name` field from the discovery response (the display name), NOT the UUID `identifier`
- Workflow JSON may use `"source_mode": "camera"` for Syphon sources — convert to `"source_mode": "syphon"` in the API call

**OpenCV dependency:** `cv2` may not be installed in the venv despite appearing in `pip list`. If `import cv2` fails, run `uv pip install opencv-python` first.

**Port conflicts:** On macOS, port 8022 may be used by Cursor IDE. If `[Errno 48] address already in use` appears in logs, use a different port (e.g., 8033). Always check with `lsof -i :<port>` before starting.

**Cloud mode recording timing:** In cloud mode, start recordings shortly after the session starts (within ~5s). If WebRTC output tracks end before recording starts, only ~30 buffered frames (~1s) will be captured.

**Debugging:**

- Check frame flow with `GET /api/v1/session/metrics`
- Check logs with `GET /api/v1/logs/tail?lines=50`
- `frames_in > 0, frames_out = 0` → pipeline processing failing
- All sinks return same frame → per-sink routing issue in HeadlessSession
- Syphon source black/missing → check logs for `"Syphon server not found"` — verify source_name matches display name from discovery

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
