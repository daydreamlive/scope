# CLAUDE.md

## Project Overview

Daydream Scope is a tool for running real-time, interactive generative AI video pipelines. It uses a Python/FastAPI backend with a React/TypeScript frontend with support for multiple autoregressive video diffusion models with WebRTC streaming. The frontend and backend are also bundled into an Electron desktop app.

### Product Direction: Interactive AI Persona

We are evolving Daydream Scope from a prompt-driven video generation tool into a **real-time interactive AI persona** system. The end goal is a persistent AI character rendered via real-time video generation that the user can **chat with** and **direct via natural language** (e.g., "stand up", "wave at the camera", "turn around", "look sad").

**Key differences from the current system:**

- **Current (LongLive pipeline):** User writes text prompts → video adapts visually in real time. One-directional creative tool.
- **Target (Persona pipeline):** User has a conversation with a visible AI character → the character responds with speech/text AND corresponding video actions. Two-directional interactive experience.

**Core requirements for the persona system:**

1. **Persistent character identity** — The generated video must maintain a consistent character appearance, outfit, and environment across the entire session. Character identity must not drift between frames or after action changes.
2. **Action/pose control** — User text commands (e.g., "wave", "sit down", "nod") must be translated into motion/pose directives that the video generation pipeline can execute. This likely requires a structured action vocabulary or an LLM-based action interpreter that maps free-text to pose/motion parameters.
3. **Conversational interface** — A chat UI where the user sends messages and receives both text responses (from an LLM) and visual responses (from the video pipeline). The LLM decides what the character says AND what physical actions to perform.
4. **Real-time streaming** — Video must remain real-time and low-latency via WebRTC, same as today. Actions should begin rendering within a frame budget, not after a long planning delay.
5. **Emotion and expression control** — Beyond gross motor actions, the persona should support facial expressions and emotional states (smile, look confused, laugh) driven by conversation context.

**Architectural implications:**

- A new **Persona pipeline** (alongside or replacing LongLive) that accepts structured action/expression directives rather than raw text prompts.
- An **action interpreter** layer (likely LLM-based) that sits between the chat interface and the video pipeline, translating conversation into `{ action, expression, dialogue }` tuples.
- A **character consistency module** — either via persistent latent conditioning, reference image anchoring, or identity-preserving fine-tuning — to prevent character drift.
- The **frontend** needs a chat interface (text input + message history) in addition to or replacing the current timeline/prompt editor.
- **Audio/TTS integration** (future) — the persona should eventually speak responses aloud, synchronized with lip movement in the generated video.

When making architectural decisions, always consider how they serve the interactive persona use case. Prefer designs that maintain character consistency, support low-latency action transitions, and cleanly separate the conversational AI layer from the video generation layer.

## Task Tracking

- **Always check TODO.md** at the root of this project before starting work — it contains the current state of all tasks, what's in progress, what's pending, and what's done.
- **Update TODO.md** as tasks are completed (`[x]`), started (`[~]`), blocked, or skipped (`[-]`).
- When new work is discovered during a session, add it as `[ ]` in the appropriate phase section.
- Keep TODO.md as the **single source of truth** for project status. Do not rely on memory or conversation history for what needs to be done next.
- Keep completed items visible for a few sessions for context, then move them to the Completed section.

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
PIPELINE=persona uv run daydream-scope   # Run with persona pipeline
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

#### Persona Pipeline (planned / in progress)

The persona pipeline extends the existing pipeline architecture with these additional components:

- **`core/pipelines/persona/`**: The persona video generation pipeline
  - Accepts structured action/expression directives (not raw user text)
  - Maintains character identity state across frames
  - Config includes character definition (appearance, reference images, voice)

- **`core/persona/`**: Persona orchestration layer (separate from pipelines)
  - **Action interpreter**: Translates free-text user messages into structured `{ action, expression, dialogue }` directives via an LLM call
  - **Conversation manager**: Maintains chat history, manages persona personality/system prompt
  - **Character state**: Tracks current pose, expression, and activity so transitions are coherent

- **`server/chat.py`** (planned): WebSocket or REST endpoint for the chat interface, connecting the conversation manager → action interpreter → pipeline manager flow

### Frontend (`frontend/src/`)

- React 19 + TypeScript + Vite
- Radix UI components with Tailwind CSS
- Timeline editor for prompt sequencing (existing, may be retained for advanced use)
- **Chat interface** (planned): Message history panel + text input for conversational interaction with the persona

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
- **Separation of concerns**: The conversational AI layer (LLM, chat, action interpretation) must be cleanly separated from the video generation layer (pipeline). The action interpreter outputs a structured schema that the pipeline consumes — they should never be tightly coupled.

### Additional Documentation

This documentation can be used to understand the architecture of the project:

- The `docs/api` directory contains server API reference
- The `docs/architecture` contains architecture documents describing different systems used within the project

## Contributing Requirements

- All commits must be signed off (DCO): `git commit -s`
- Pre-commit hooks run ruff (Python) and prettier/eslint (frontend)
- Models stored in `~/.daydream-scope/models` (configurable via `DAYDREAM_SCOPE_MODELS_DIR`)

## Style Guidelines

### Backend

- Use relative imports if it is single or double dot (eg .package or ..package) and otherwise use an absolute import
- `scope.server` can import from `scope.core`, but `scope.core` must never import from `scope.server`
- The persona orchestration layer (`scope.core.persona`) can import from `scope.core.pipelines` but not from `scope.server`

## Verifying Work

Follow these guidelines for verifying work when implementation for a task is complete.

### Backend

- Use `uv run daydream-scope` to confirm that the server starts up without errors.

### Frontend

- Use `npm run build` to confirm that builds work properly.
