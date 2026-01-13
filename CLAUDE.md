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

## Contributing Requirements

- All commits must be signed off (DCO): `git commit -s`
- Pre-commit hooks run ruff (Python) and prettier/eslint (frontend)
- Models stored in `~/.daydream-scope/models` (configurable via `DAYDREAM_SCOPE_MODELS_DIR`)

## Style Guidelines

### Backend

- Use relative imports if it is single or double dot (eg .package or ..package) and otherwise use an absolute import
- `scope.server` can import from `scope.core`, but `scope.core` must never import from `scope.server`

## Verifying Work

Follow these guidelines for verifing work when implementation for a task is complete.

### Backend

- Use `uv run daydream-scope` to confirm that the server starts up without errors.

### Frontend

- Use `npm run build` to confirm that builds work properly.
