# Plugin Architecture

## Overview

The Scope plugin system enables third-party extensions to provide custom pipelines. This document describes the architectural design and data flows that enable plugin discovery, installation, and lifecycle management.

### Key Technologies

- **pluggy**: Python hook system for pipeline registration and discovery
- **uv**: Fast Python package manager for dependency resolution and installation
- **Electron IPC**: Communication bridge between desktop app and frontend

### Architecture Layers

```
Desktop App (Electron) → Frontend (React) → Backend (FastAPI) → Plugins
```

Each layer has distinct responsibilities, communicating through well-defined interfaces.

---

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Desktop App (Electron)                    │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Deep Links  │───▶│  IPC Bridge  │◀──▶│ Python Process    │  │
│  │ File Browse │    │              │    │ Manager           │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Settings UI │───▶│  API Client  │───▶│ State Management  │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  REST API   │───▶│Plugin Manager│───▶│ Pipeline Registry │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│                              │                                   │
│                     ┌────────┴────────┐                         │
│                     ▼                 ▼                         │
│              ┌────────────┐    ┌────────────┐                   │
│              │ Dependency │    │   Venv     │                   │
│              │ Validator  │    │ Snapshot   │                   │
│              └────────────┘    └────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Plugin Discovery

Plugins integrate with Scope through Python's entry point mechanism:

1. Plugins declare entry points in their `pyproject.toml` under `[project.entry-points."scope"]`
2. The backend uses pluggy hooks to discover installed plugins at startup
3. Each plugin implements a `register_pipelines` hook to register its pipeline implementations
4. The Pipeline Registry maintains a mapping of pipeline IDs to their implementations

### Plugin Sources

The system supports three installation sources:

| Source | Description |
|--------|-------------|
| **PyPI** | Standard Python packages |
| **Git** | Direct repository installation |
| **Local** | File system paths |

Local installations support editable mode for rapid development iteration. Git installation is currently the easiest path for installing third-party plugins.

---

## Core Flows

### Plugin Installation Flow

```
┌──────┐     ┌──────────┐     ┌─────────┐     ┌─────────────┐
│ User │     │ Frontend │     │ Backend │     │ Desktop App │
└──┬───┘     └────┬─────┘     └────┬────┘     └──────┬──────┘
   │              │                │                  │
   │ Click Install│                │                  │
   │─────────────▶│                │                  │
   │              │ POST /plugins  │                  │
   │              │───────────────▶│                  │
   │              │                │ Validate deps    │
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │                │ Capture venv     │
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │                │ Install via uv   │
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │     200 OK     │                  │
   │              │◀───────────────│                  │
   │              │ Request restart│                  │
   │              │───────────────────────────────────▶
   │              │                │                  │ Respawn server
   │              │                │                  │────────┐
   │              │                │                  │◀───────┘
   │              │ Poll health    │                  │
   │              │───────────────▶│                  │
   │              │ Refresh pipelines                 │
   │              │───────────────▶│                  │
   │              │                │                  │
```

**Step by step:**

1. User initiates install (UI or deep link)
2. Frontend sends install request to backend API
3. Backend validates dependencies won't conflict with existing environment
4. Backend captures current venv state (for rollback)
5. Backend resolves and installs dependencies via uv
6. Backend updates plugin registry
7. Frontend triggers server restart
8. Server restarts with new plugin loaded
9. Frontend polls until server is healthy
10. Frontend refreshes pipeline list

**Rollback**: If installation fails at any step, the venv is restored to its captured state.

### Plugin Update Flow

The update flow has two phases: **detection** (happens automatically when plugins are listed) and **execution** (triggered by the user). The execution phase reuses the installation flow with an `upgrade: true` flag.

#### Update Detection

```
┌──────────┐     ┌─────────┐
│ Frontend │     │ Backend │
└────┬─────┘     └────┬────┘
     │                │
     │ GET /plugins   │
     │───────────────▶│
     │                │ For each plugin:
     │                │ _check_plugin_update()
     │                │────────┐
     │                │        │ Compare installed
     │                │        │ vs latest version
     │                │◀───────┘
     │  Plugin list   │
     │  (with         │
     │  update_available│
     │  flags)        │
     │◀───────────────│
     │                │
```

**Step by step:**

1. Frontend fetches the plugin list via `GET /plugins`
2. Backend iterates over installed plugins and calls `_check_plugin_update()` for each
3. For PyPI plugins, the installed version is compared against the latest version on PyPI
4. For Git plugins, the installed commit hash is compared against the latest commit on the remote
5. Local plugins are skipped (use Reload instead)
6. Each plugin in the response includes an `update_available` flag
7. Frontend displays an "Update available" badge on plugins where the flag is `true`

#### Update Execution

```
┌──────┐     ┌──────────┐     ┌─────────┐     ┌─────────────┐
│ User │     │ Frontend │     │ Backend │     │ Desktop App │
└──┬───┘     └────┬─────┘     └────┬────┘     └──────┬──────┘
   │              │                │                  │
   │ Click Update │                │                  │
   │─────────────▶│                │                  │
   │              │ POST /plugins  │                  │
   │              │ {upgrade: true}│                  │
   │              │───────────────▶│                  │
   │              │                │ Capture venv     │
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │                │ Compile with     │
   │              │                │ --upgrade-package│
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │                │ Sync deps        │
   │              │                │────────┐         │
   │              │                │◀───────┘         │
   │              │     200 OK     │                  │
   │              │◀───────────────│                  │
   │              │ Request restart│                  │
   │              │───────────────────────────────────▶
   │              │                │                  │ Respawn server
   │              │                │                  │────────┐
   │              │                │                  │◀───────┘
   │              │ Poll health    │                  │
   │              │───────────────▶│                  │
   │              │ Refresh pipelines                 │
   │              │───────────────▶│                  │
   │              │                │                  │
```

**Step by step:**

1. User clicks the Update button on a plugin
2. Frontend sends `POST /plugins` with the plugin spec and `upgrade: true`
3. Backend captures the current venv state (for rollback)
4. Backend runs `uv pip compile` with `--upgrade-package` targeting only the plugin package
5. Backend syncs the environment with the newly resolved dependencies
6. Backend updates the plugin registry
7. Frontend triggers server restart
8. Server restarts with the updated plugin loaded
9. Frontend polls until server is healthy
10. Frontend refreshes pipeline list

**Rollback**: If the update fails at any step, the venv is restored to its pre-update state, just like a failed installation.

#### Source-Specific Update Behavior

| Source | Detection Method | Notes |
|--------|-----------------|-------|
| **PyPI** | Compares installed version against latest version on PyPI | Standard version comparison |
| **Git** | Compares installed commit hash against latest remote commit | Detects new commits on the default branch |
| **Local** | Skipped | Local plugins use [Reload](#manual-reload-flow-editable-plugins-only) instead |

### Plugin Uninstallation Flow

1. User initiates uninstall
2. Backend unloads any active pipelines from the plugin
3. Backend removes plugin from registry
4. Backend uninstalls package via uv
5. Frontend triggers server restart
6. Frontend refreshes pipeline list

### Manual Reload Flow (Editable Plugins Only)

For local/editable plugins, developers can trigger a reload to pick up code changes:

```
┌───────────┐     ┌──────────┐     ┌─────────┐     ┌─────────────┐
│ Developer │     │ Frontend │     │ Backend │     │ Desktop App │
└─────┬─────┘     └────┬─────┘     └────┬────┘     └──────┬──────┘
      │                │                │                  │
      │ Modify code    │                │                  │
      │────────┐       │                │                  │
      │◀───────┘       │                │                  │
      │ Click Reload   │                │                  │
      │───────────────▶│                │                  │
      │                │ Request restart│                  │
      │                │───────────────────────────────────▶
      │                │                │                  │ Respawn server
      │                │                │                  │────────┐
      │                │                │                  │◀───────┘
      │                │ Poll health    │                  │
      │                │───────────────▶│                  │
      │                │ Refresh pipelines                 │
      │                │───────────────▶│                  │
      │                │                │                  │
```

This triggers a server restart to ensure Python modules are fully reloaded.

In standalone mode (without the desktop app), the reload flow is the same except the server performs a self-restart instead of being respawned by the desktop app (see [Standalone Mode](#standalone-mode)).

### Deep Link Installation Flow

External sources can facilitate plugin installation via protocol URLs:

```
daydream-scope://install-plugin?package=<spec>
```

**Package Spec Format:**

The `<spec>` parameter must be URL encoded. Examples:

| Source | Raw Spec | Encoded URL |
|--------|----------|-------------|
| PyPI | `my-plugin` | `daydream-scope://install-plugin?package=my-plugin` |
| Git | `git+https://github.com/user/repo.git` | `daydream-scope://install-plugin?package=git%2Bhttps%3A%2F%2Fgithub.com%2Fuser%2Frepo.git` |

**Flow:**

1. External source opens the deep link URL
2. Desktop app receives URL via OS protocol handler
3. If app is starting: stores pending deep link for later processing
4. Once frontend is loaded: sends action via IPC to renderer
5. Frontend opens settings with plugin tab and pre-filled package spec
6. User confirms installation → standard installation flow begins

---

## Component Responsibilities

### Backend Components

| Component | Responsibility |
|-----------|----------------|
| **Plugin Manager** | Singleton that manages plugin lifecycle (install, uninstall, update, reload) |
| **Dependency Validator** | Pre-validates that new packages won't break existing environment |
| **Venv Snapshot** | Captures and restores environment state for safe rollback |
| **Pipeline Registry** | Maps pipeline IDs to their implementations and source plugins |
| **REST API** | Exposes plugin operations to frontend |

### Frontend Components

| Component | Responsibility |
|-----------|----------------|
| **Settings Dialog** | User interface for plugin management |
| **API Client** | HTTP calls to backend plugin endpoints |
| **Restart Coordinator** | Handles server restart and health polling |
| **Pipeline Context** | Refreshes available pipelines after changes |

### Desktop App Components

| Component | Responsibility |
|-----------|----------------|
| **Python Process Manager** | Spawns/respawns backend server, handles restart signals |
| **Deep Link Handler** | Receives and parses protocol URLs |
| **IPC Bridge** | Communicates between main process and renderer |
| **File Browser** | Native dialog for selecting local plugin directories |

> **Note:** The File Browser is a desktop convenience feature. In standalone mode, users can type local paths directly into the plugin installation input field.

---

## Server Restart Protocol

The restart protocol differs depending on how the server is running.

### Managed Mode (Desktop App)

When running in the desktop app, server restarts are handled automatically:

```
┌──────────┐     ┌─────────┐     ┌─────────────┐
│ Frontend │     │ Backend │     │ Desktop App │
└────┬─────┘     └────┬────┘     └──────┬──────┘
     │                │                  │
     │ POST /restart  │                  │
     │───────────────▶│                  │
     │                │ Exit code 42     │
     │                │─────────────────▶│
     │                │                  │ Wait for port
     │                │                  │────────┐
     │                │                  │◀───────┘
     │                │                  │ Respawn server
     │                │◀─────────────────│
     │ Poll /health   │                  │
     │───────────────▶│                  │
     │     200 OK     │                  │
     │◀───────────────│                  │
     │ Resume ops     │                  │
     │────────┐       │                  │
     │◀───────┘       │                  │
```

**Key points:**

- Exit code 42 signals intentional restart (not a crash)
- Brief wait ensures the port is released before respawn
- Frontend polls health endpoint until server is ready
- This ensures proper Python module reloading since imports are cached

### Standalone Mode

When running the server directly (e.g., `uv run daydream-scope`), the server performs a self-restart:

- **Unix/macOS**: Uses `os.execv()` to replace the current process in-place
- **Windows**: Spawns a new subprocess and exits the old one

This achieves the same result as managed mode—full Python module cache refresh—without requiring an external process manager.

---

## Data Storage

Plugin state is persisted in the user's data directory:

| Data | Location | Purpose |
|------|----------|---------|
| Plugin list | `~/.daydream-scope/plugins/plugins.txt` | Installed package specs |
| Resolved deps | `~/.daydream-scope/plugins/resolved.txt` | Lock file for reproducibility; baseline for update detection |
| Venv backup | `~/.daydream-scope/plugins/freeze.txt` | Rollback state |

---

## Creating a Plugin

### Requirements

Plugins must:

1. Define an entry point in `pyproject.toml`
2. Implement the `register_pipelines` hook
3. Provide Pipeline subclasses with configuration schemas

### Entry Point Configuration

```toml
[project.entry-points."scope"]
my_plugin = "my_plugin.plugin"
```

### Hook Implementation

```python
from scope.core.pipelines.hookspecs import hookimpl

@hookimpl
def register_pipelines(register):
    from .pipelines import MyCustomPipeline
    register(MyCustomPipeline)
```

### Pipeline Implementation

Pipelines must:

- Inherit from the base `Pipeline` class
- Define a Pydantic configuration schema
- Implement the `__call__()` method for frame generation

---

## Error Handling Strategy

The plugin system uses defensive error handling at each stage:

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| **Dependency conflicts** | Dry-run compilation before install | Installation blocked with clear error message |
| **Installation failures** | Exception during uv install | Venv rolled back to pre-install state |
| **Runtime errors** | Pipeline execution failure | Pipeline unloaded without affecting others |
| **Network errors** | Health check timeouts | Frontend retries with exponential backoff |

This multi-layer approach ensures that plugin operations cannot corrupt the base Scope installation.
