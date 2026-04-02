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

**Known issue — `_wire_cloud_outputs` attribute paths:**

The `_wire_cloud_outputs` function in `mcp_router.py` wires cloud WebRTC output tracks to per-sink and per-record queues. Key attribute paths that were recently fixed:

- Sink queues: `frame_processor.sink_manager._sink_queues_by_node` (NOT `frame_processor._sink_queues_by_node`)
- Cloud relay callback: `frame_processor._cloud_relay.on_frame_from_cloud` (NOT `frame_processor._on_frame_from_cloud`)
- Output handlers: `webrtc_client.output_handlers[i]` (list, NOT dict `.get(i)`)

**Known issue — `alias_pipeline` collision:**

When a graph node ID collides with an existing pipeline ID (e.g., node `passthrough` with `pipeline_id: "split-screen"` when a builtin `passthrough` pipeline exists), `pipeline_manager.alias_pipeline()` must overwrite the existing entry. The old code returned early if the alias key already existed.

## MCP Server Testing with Local Cloud Dev

When asked to test Scope via MCP tools (e.g., with a workflow JSON), follow this sequence directly — do not read source code to figure out the API.

**Setup (run both as background processes):**

First, kill any existing processes on the ports to avoid "address already in use" errors:

```bash
lsof -ti:8002 -ti:8022 | xargs kill -9 2>/dev/null
```

Then start the instances (start cloud first, wait for it to be healthy before starting local):

```bash
# Cloud instance:
SCOPE_CLOUD_WS=1 uv run daydream-scope --port 8002

# Local instance (start after cloud is healthy):
SCOPE_CLOUD_WS_URL=ws://localhost:8002/ws SCOPE_CLOUD_APP_ID=local uv run daydream-scope --port 8022
```

Wait for both to be healthy: `curl -s http://localhost:8002/health && curl -s http://localhost:8022/health`

Note: The health endpoint is `/health`, not `/api/v1/health`.

**MCP tool sequence:**

1. `connect_to_scope(port=8022)` — connect to the local instance
2. `connect_to_cloud()` — connects to cloud (env vars provide app_id/url)
3. `resolve_workflow(workflow_json=...)` — validate dependencies
4. `load_pipeline(pipeline_id=...)` — load the pipeline(s) from the workflow; wait a few seconds for loading to complete. The load is proxied to the cloud instance automatically.
5. `start_stream(graph=...)` — start a headless session. Accepts a `graph` dict with nodes/edges for multi-source/multi-sink workflows, or `pipeline_id` for single-pipeline mode.
6. `capture_frame()` — capture output frame (returns file_path to a JPEG)
7. `start_recording()` / `stop_recording()` / `download_recording()` — record and download MP4

**Key details:**

- The headless session endpoint (`/api/v1/session/start`) accepts `pipeline_id` OR `graph`, plus `input_mode`, `prompts`, `input_source`
- For multi-source/multi-sink workflows, pass a `graph` dict with nodes and edges (see `graph_schema.py:GraphConfig`)
- When parsing a workflow JSON, extract unique `pipeline_id` values from `pipelines[]` for `load_pipeline`
- `resolve_workflow` takes the full workflow JSON as a **string**
- `capture_frame` and `download_recording` return file paths — use the Read tool to view the captured image
- For multi-sink graphs, `capture_frame` accepts `sink_node_id` to capture from a specific sink
- The health endpoint is `/health` (not `/api/v1/health`)

**Session runs on the LOCAL instance (port 8022), not cloud:**

The headless session (`/api/v1/session/start`, `/session/frame`, `/session/stop`, and `/recordings/headless/*`) is created on the LOCAL instance (port 8022). The local instance's FrameProcessor runs in cloud-relay mode: it sends input frames to cloud via WebRTC, receives processed output frames back, and stores them in per-sink queues. Call ALL session/frame/recording endpoints on port 8022 (the local instance), not 8002.

**Passthrough pipeline requires video input:**

The `passthrough` pipeline passes input through unchanged — it does **not** generate frames on its own. In headless mode, you must provide an `input_source`. Create a test video and pass it:

```bash
# Create a 60-second blue test video:
ffmpeg -y -f lavfi -i "color=c=blue:size=512x512:rate=30:duration=60" -c:v libx264 -pix_fmt yuv420p /tmp/test_input.mp4
```

Then start the session with `input_mode: "video"` and `input_source`:

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

**Multi-source/multi-sink graph session (video file inputs):**

When testing a workflow with multiple sources and video file inputs, pass a full graph to `/api/v1/session/start`. Source nodes should use `source_mode: "video_file"` with the file path as `source_name`. Include record nodes for recording support. Example:

```json
{
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

Create test videos with OpenCV (ffmpeg not available in this env):

```python
uv run python -c "
import cv2, numpy as np
for name, color in [('test', (0,0,255)), ('test1', (0,255,0)), ('test2', (255,0,0))]:
    w = cv2.VideoWriter(f'/tmp/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
    frame = np.zeros((512,512,3), dtype=np.uint8); frame[:] = color
    for _ in range(300): w.write(frame)
    w.release()
"
```

**CUDA OOM with local cloud dev:**

Both instances share the same GPU. For lightweight pipelines (split-screen, passthrough), run with `CUDA_VISIBLE_DEVICES=""` to force CPU mode and avoid OOM.

**Debugging frame flow:**

- Check frame flow with `/api/v1/session/metrics` on both instances
- `frames_to_cloud` > 0, `frames_from_cloud` = 0 → cloud is not sending output back; check cloud logs for pipeline errors
- Cloud session `frames_in` > 0, `frames_out` = 0 → pipeline processing failing; check cloud errors with `/api/v1/logs/tail?lines=50`
- Check `/api/v1/logs/tail?lines=50` for `ERROR` entries on both instances
- The MCP server disconnects when Scope processes are restarted; use HTTP API fallback below

**HTTP API fallback (when MCP tools are unavailable):**

If the MCP server disconnects (e.g., after restarting Scope processes), use direct HTTP calls:

| Operation | HTTP API |
|-----------|----------|
| Connect to cloud | `POST /api/v1/cloud/connect` body: `{"app_id": "local"}` |
| Cloud status | `GET /api/v1/cloud/status` |
| Resolve workflow | `POST /api/v1/workflow/resolve` body: `{"pipelines": [...]}` (pipelines array directly, **not** wrapped in `workflow_json`) |
| Load pipeline | `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["name"]}` (array, **not** `pipeline_id`) |
| Pipeline status | `GET /api/v1/pipeline/status` |
| Start session | `POST /api/v1/session/start` body: `{"pipeline_id": "name", ...}` or `{"graph": {...}}` |
| Capture frame | `GET /api/v1/session/frame` or `?sink_node_id=output` (returns JPEG binary) |
| Stream MPEG-TS | `GET /api/v1/session/output.ts` (streams `video/mp2t`; includes audio when pipeline produces it) |
| Stop session | `POST /api/v1/session/stop` |
| Start recording | `POST /api/v1/recordings/headless/start` |
| Stop recording | `POST /api/v1/recordings/headless/stop` |
| Download recording | `GET /api/v1/recordings/headless` (returns MP4 binary) |
| Logs | `GET /api/v1/logs/tail?lines=30` (not `/api/v1/logs`) |

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
