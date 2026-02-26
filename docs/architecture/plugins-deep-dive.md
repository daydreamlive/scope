# How Plugins Work in Scope

This document explains how the Scope plugin system works end-to-end, with a focus on how plugin settings (sliders, toggles, dropdowns) propagate from the backend to the frontend in both **local mode** and **cloud relay mode**.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Plugin Discovery and Registration](#plugin-discovery-and-registration)
- [Schema Generation: How Settings Are Defined](#schema-generation-how-settings-are-defined)
- [Local Mode: Settings Propagation Flow](#local-mode-settings-propagation-flow)
- [Cloud Relay Mode: Settings Propagation Flow](#cloud-relay-mode-settings-propagation-flow)
- [Frontend: From JSON Schema to UI Widgets](#frontend-from-json-schema-to-ui-widgets)
- [Key Differences: Local vs Cloud Relay](#key-differences-local-vs-cloud-relay)

---

## Architecture Overview

The plugin system has four layers:

```
+------------------------------------------------------+
|  Desktop App (Electron)                              |
|  - Python process lifecycle                          |
|  - Deep link handling (daydream-scope:// protocol)   |
+------------------------------------------------------+
         |
+------------------------------------------------------+
|  Frontend (React/TypeScript)                         |
|  - Plugin management UI (Settings dialog)            |
|  - Schema-driven settings rendering                  |
|  - Cloud-aware API routing (useApi hook)             |
+------------------------------------------------------+
         |
+------------------------------------------------------+
|  Backend (FastAPI / Python)                          |
|  - REST endpoints for plugin CRUD + pipeline schemas |
|  - @cloud_proxy() decorator for transparent relay    |
|  - PipelineManager for lifecycle                     |
+------------------------------------------------------+
         |
+------------------------------------------------------+
|  Plugin Layer                                        |
|  - Entry point discovery (pluggy + setuptools)       |
|  - Pipeline implementations with Pydantic schemas    |
|  - register_pipelines hook                           |
+------------------------------------------------------+
```

---

## Plugin Discovery and Registration

### How Plugins Are Found

Scope uses Python's **entry point** mechanism combined with the **pluggy** hook system. A plugin declares itself in its `pyproject.toml`:

```toml
[project.entry-points."scope"]
my_plugin = "my_plugin.plugin"
```

### Registration Flow at Startup

When the Scope backend starts, the registry module (`src/scope/core/pipelines/registry.py`) auto-initializes:

```
registry.py import
    |
    v
_initialize_registry()
    |
    +---> _register_pipelines()          # 1. Register built-in pipelines
    |         |
    |         +---> For each built-in pipeline:
    |                  importlib.import_module(module_path)
    |                  Check GPU VRAM requirements
    |                  PipelineRegistry.register(pipeline_id, pipeline_class)
    |
    +---> ensure_plugins_installed()     # 2. Re-install from resolved.txt if missing
    |
    +---> load_plugins()                 # 3. Discover plugins via entry points
    |         |
    |         +---> pluggy.PluginManager.load_setuptools_entrypoints("scope")
    |                  Scans all installed packages for "scope" entry points
    |
    +---> register_plugin_pipelines()    # 4. Call hook on each plugin
              |
              +---> pm.hook.register_pipelines(register=callback)
                       Each plugin's register_pipelines() is called
                       callback(PipelineClass) adds to PipelineRegistry
```

The `register_pipelines` hook (defined in `src/scope/core/plugins/hookspecs.py`) is the contract:

```python
# In the plugin's __init__.py or plugin.py:
@hookimpl
def register_pipelines(register):
    register(MyPipeline)
```

### What Gets Registered

Each pipeline class provides:
- A `Pipeline` implementation (the `__call__` method that processes frames)
- A `BasePipelineConfig` subclass (Pydantic model defining all settings)

The config class is the key to the entire settings system -- it is the **single source of truth** for what parameters a pipeline exposes and how they should appear in the UI.

---

## Schema Generation: How Settings Are Defined

### Defining a Setting in the Backend

Pipeline settings are Pydantic fields on a `BasePipelineConfig` subclass (`src/scope/core/pipelines/base_schema.py`). Each field maps to a UI widget based on its type and constraints:

```python
class MyPipelineConfig(BasePipelineConfig):
    pipeline_id: ClassVar[str] = "my-pipeline"
    pipeline_name: ClassVar[str] = "My Pipeline"

    # This becomes a SLIDER (float + min/max)
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Effect intensity",
        json_schema_extra=ui_field_config(order=1),
    )

    # This becomes a TOGGLE (bool)
    enabled: bool = Field(
        default=True,
        description="Enable the effect",
        json_schema_extra=ui_field_config(order=2),
    )

    # This becomes a DROPDOWN (enum)
    mode: MyEnum = Field(
        default=MyEnum.FAST,
        description="Processing mode",
        json_schema_extra=ui_field_config(order=3),
    )
```

### The `ui_field_config()` Helper

The `ui_field_config()` function (`base_schema.py:101-147`) builds the `json_schema_extra` dict that Pydantic embeds into the JSON schema under a `"ui"` key:

```python
def ui_field_config(
    *,
    order: int | None = None,        # Display order (lower = first)
    component: str | None = None,     # Complex component name (e.g., "vace", "lora")
    modes: list[str] | None = None,   # Restrict to input modes (e.g., ["video"])
    is_load_param: bool = False,      # True = disabled during streaming
    label: str | None = None,         # Override label (description becomes tooltip)
    category: Literal["configuration", "input"] | None = None,
) -> dict[str, Any]:
```

Fields **without** `json_schema_extra` (no `ui` key) are **not rendered** in the UI -- this is how internal/base fields like `height`, `width`, `manage_cache` are hidden from the settings panel unless a pipeline explicitly opts them in.

### Schema Serialization

When the API is called, `get_schema_with_metadata()` (`base_schema.py:347-396`) combines:

1. **Pipeline metadata** - id, name, description, version, feature flags
2. **Mode information** - supported modes, default mode, mode-specific overrides
3. **JSON schema** - `cls.model_json_schema()` (Pydantic's built-in serializer)

The resulting JSON for a slider field looks like:

```json
{
  "properties": {
    "intensity": {
      "type": "number",
      "default": 0.5,
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Effect intensity",
      "ui": {
        "category": "configuration",
        "is_load_param": false,
        "order": 1
      }
    }
  }
}
```

---

## Local Mode: Settings Propagation Flow

In local mode, the frontend talks directly to the local FastAPI backend over HTTP.

### Diagram

```
 Frontend (React)                         Backend (FastAPI, localhost:8000)
 ================                         ==================================

 usePipelines() hook
      |
      |  GET /api/v1/pipelines/schemas
      |------------------------------->   get_pipeline_schemas()
      |                                        |
      |                                        |  PipelineRegistry.list_pipelines()
      |                                        |  For each pipeline:
      |                                        |    config_class = get_config_class()
      |                                        |    schema = config_class.get_schema_with_metadata()
      |                                        |      |
      |                                        |      +-> Pydantic model_json_schema()
      |                                        |      +-> Pipeline metadata (name, modes, flags)
      |                                        |      +-> Mode-specific defaults
      |                                        |    schema["plugin_name"] = plugin_manager.get_plugin_for_pipeline()
      |                                        |
      |   <-- PipelineSchemasResponse ---------|
      |   {
      |     "pipelines": {
      |       "my-pipeline": {
      |         "name": "My Pipeline",
      |         "config_schema": { ... },
      |         "plugin_name": "my-scope-plugin",
      |         ...
      |       }
      |     }
      |   }
      |
      v
 transformSchemas()
      |  Converts to PipelineInfo objects
      |  Extracts configSchema, feature flags, etc.
      v
 SettingsPanel renders
      |
      +-> parseConfigurationFields(configSchema, inputMode)
      |     |
      |     +-> Filter: only fields with "ui" key in json_schema_extra
      |     +-> Filter: by category ("configuration" vs "input")
      |     +-> Filter: by input mode (text/video)
      |     +-> Sort: by ui.order, then alphabetically
      |
      +-> inferPrimitiveFieldType(prop)
      |     |
      |     +-> enum in prop?          --> "enum"    --> DropdownField
      |     +-> type === "boolean"?    --> "toggle"  --> ToggleField
      |     +-> type === "number" &&
      |        min+max defined?        --> "slider"  --> SliderField
      |     +-> type === "number"?     --> "number"  --> NumberField
      |     +-> type === "string"?     --> "text"    --> TextField
      |
      +-> Render <SliderField>, <ToggleField>, <DropdownField>, etc.
            Props: min/max from schema, default from schema, step inferred from type
```

### Key Files (Local Mode)

| Step | File | Key Function |
|------|------|-------------|
| API endpoint | `src/scope/server/app.py:550-588` | `get_pipeline_schemas()` |
| Schema generation | `src/scope/core/pipelines/base_schema.py:347-396` | `get_schema_with_metadata()` |
| UI metadata helper | `src/scope/core/pipelines/base_schema.py:101-147` | `ui_field_config()` |
| Registry | `src/scope/core/pipelines/registry.py:76-88` | `get_config_class()` |
| Plugin discovery | `src/scope/core/plugins/manager.py:91-110` | `PluginManager` |
| Frontend API call | `frontend/src/lib/api.ts:580-596` | `getPipelineSchemas()` |
| Frontend hook | `frontend/src/hooks/usePipelines.ts:66-89` | `refreshPipelines()` |
| Schema parsing | `frontend/src/lib/schemaSettings.ts:125-130` | `parseConfigurationFields()` |
| Widget inference | `frontend/src/lib/schemaSettings.ts:60-97` | `inferPrimitiveFieldType()` |
| Slider rendering | `frontend/src/components/PrimitiveFields.tsx:175-215` | `SliderField` |

---

## Cloud Relay Mode: Settings Propagation Flow

In cloud relay mode, the Scope backend runs on a fal.ai GPU instance. The frontend connects via a **single persistent WebSocket** to a `ScopeApp` running on fal, which proxies API requests to the Scope backend on localhost within the same container.

### Diagram

```
 Frontend (React)                    fal.ai Container
 ================                    ===================================================
                                     ScopeApp (fal_app.py)          Scope Backend
                                     WebSocket handler              (localhost:8000)
                                     =====================          ================

 useCloudAdapter()
      |
      |  wss://fal-url/ws?fal_jwt_token=...
      |----WebSocket Connect--------->
      |                                 |
      |  <-- {"type":"ready"} ---------|
      |
 usePipelines() hook
      |
      |  isCloudMode && adapter?
      |  --> adapter.api.getPipelineSchemas()
      |
      |  WS send:                       WS receive:
      |  {                              handle_api_request()
      |    "type": "api",                    |
      |    "method": "GET",                  |  httpx.get("http://localhost:8000
      |    "path": "/api/v1/                 |             /api/v1/pipelines/schemas")
      |             pipelines/schemas",      |
      |    "request_id": "req_1_..."         |-------->   get_pipeline_schemas()
      |  }                                   |                  |
      |                                      |                  |  (Same local logic
      |                                      |                  |   as above -- iterates
      |                                      |                  |   registry, generates
      |                                      |                  |   schemas from Pydantic)
      |                                      |                  |
      |                                      |  <--- JSON ------|
      |                                      |
      |  WS receive:                    WS send:
      |  {                              {
      |    "type": "api_response",        "type": "api_response",
      |    "request_id": "req_1_...",     "request_id": "req_1_...",
      |    "status": 200,                 "status": 200,
      |    "data": {                      "data": { ... }
      |      "pipelines": { ... }       }
      |    }
      |  }
      |
      v
 (Same frontend logic from here: transformSchemas -> parse -> render)
```

### The `@cloud_proxy()` Decorator

The `/api/v1/pipelines/schemas` endpoint uses the `@cloud_proxy()` decorator (`src/scope/server/cloud_proxy.py:201-257`). This creates a **dual-mode endpoint**:

```python
@app.get("/api/v1/pipelines/schemas")
@cloud_proxy()                                   # <-- The magic
async def get_pipeline_schemas(
    http_request: Request,
    cloud_manager: CloudConnectionManager = Depends(...),
):
    # This body only runs in local mode (cloud_manager not connected).
    # In cloud relay mode, the decorator intercepts and proxies to cloud.
    ...
```

When the frontend is running locally but connected to a cloud backend:
1. Frontend calls `GET /api/v1/pipelines/schemas` on the **local** backend
2. `@cloud_proxy()` checks `cloud_manager.is_connected`
3. If connected, it **proxies** the request to the cloud over the WebSocket relay
4. The cloud's Scope backend runs the same handler **locally** (no decorator intercept, since `cloud_manager` there is not connected)
5. Response travels back: cloud -> WebSocket -> local backend -> frontend

When the frontend is running in the browser connecting directly to the cloud (e.g., the web app):
1. Frontend connects via `CloudAdapter` WebSocket directly to fal
2. `adapter.api.getPipelineSchemas()` sends a WebSocket message
3. `fal_app.py` receives it in `handle_api_request()` and calls `httpx.get("http://localhost:8000/api/v1/pipelines/schemas")`
4. The Scope backend running in the same container handles it locally
5. Response travels back through WebSocket to the frontend

### Built-In Plugins in Cloud Mode

Since plugin **installation is blocked** in cloud mode (security: prevents arbitrary code execution on shared GPU instances), plugins must be **pre-baked** into the Docker image:

```dockerfile
# Dockerfile (lines 49-72)
ARG INSTALL_CLOUD_PLUGINS=false
COPY cloud-plugins.txt* ./
RUN if [ "$INSTALL_CLOUD_PLUGINS" = "true" ] && [ -f cloud-plugins.txt ]; then \
      grep -v '^#' cloud-plugins.txt | ... | while read -r plugin; do \
        uv pip install --torch-backend cu128 "$plugin"; \
      done; \
    fi
```

The `cloud-plugins.txt` file is generated by `scripts/generate_cloud_plugins.py`, which fetches popular plugins from the Daydream API:

```
git+https://github.com/user/plugin-a.git  # Plugin A
git+https://github.com/user/plugin-b.git  # Plugin B
```

At container startup, these pre-installed plugins are discovered via the same entry point mechanism -- they appear in the registry just like locally installed plugins. The schemas endpoint returns them alongside built-in pipelines, and the frontend renders their settings identically.

### Plugin Installation Blocking

In `fal_app.py`, plugin installation is explicitly blocked:

```python
# fal_app.py:615-622
if method == "POST" and path == "/api/v1/plugins":
    return {
        "type": "api_response",
        "request_id": request_id,
        "status": 403,
        "error": "Plugin installation is not available in cloud mode",
    }
```

This is a security measure: plugins can execute arbitrary Python code, which is dangerous in a multi-tenant cloud environment.

---

## Frontend: From JSON Schema to UI Widgets

The frontend uses a **schema-first** approach -- no UI code exists in the backend. The backend only defines constraints and metadata; the frontend infers widget types.

### Widget Selection Rules

Defined in `frontend/src/lib/schemaSettings.ts:60-97`:

| Schema Property | Widget | Example |
|----------------|--------|---------|
| `type: "boolean"` | Toggle switch | `enabled: bool` |
| `type: "number"` + `minimum` + `maximum` | Slider | `intensity: float = Field(ge=0.0, le=1.0)` |
| `type: "number"` (no min/max) | Number input (+/- buttons) | `seed: int` |
| `type: "string"` | Text input | `prompt: str` |
| `enum` values present | Dropdown | `mode: MyEnum` |
| `ui.component` is a complex name | Specialized component | `"vace"`, `"lora"`, `"resolution"` |

### Field Categories

Fields are split into two panels based on `ui.category`:

- **`"configuration"`** (default) -- Shown in the **Settings panel**. These are pipeline parameters like strength, steps, etc.
- **`"input"`** -- Shown in the **Input & Controls panel** below Prompts. These are per-frame controls.

### Load vs Runtime Parameters

The `ui.is_load_param` flag controls editability during streaming:

- **`is_load_param: true`** -- Passed to `__init__()` when loading the pipeline. **Disabled** while streaming (requires pipeline reload to change).
- **`is_load_param: false`** (default) -- Passed via `kwargs` to `__call__()` on every frame. **Editable** during streaming for real-time tweaking.

---

## Key Differences: Local vs Cloud Relay

| Aspect | Local Mode | Cloud Relay Mode |
|--------|-----------|-----------------|
| **Transport** | Direct HTTP (`fetch`) | WebSocket through `CloudAdapter` to `fal_app.py`, then `httpx` to `localhost:8000` |
| **Schema source** | Local `PipelineRegistry` | Remote `PipelineRegistry` on fal.ai GPU container |
| **Plugin installation** | Allowed (POST `/api/v1/plugins`) | Blocked (403 Forbidden) |
| **Plugin availability** | User installs via Settings UI | Pre-baked into Docker image at build time |
| **Plugin updates** | Auto-detected, update via UI | Requires Docker image rebuild |
| **Plugin reload** | Supported for local/editable plugins | Not available |
| **Settings rendering** | Identical -- same schema, same frontend code | Identical -- same schema, same frontend code |
| **API routing** | `useApi` hook -> direct `fetch()` | `useApi` hook -> `adapter.api.*` -> WebSocket |
| **`@cloud_proxy()`** | Skipped (not connected) | Intercepts and proxies to cloud |

The key insight is that **the settings JSON schema is identical regardless of mode**. The frontend renders the exact same widgets whether the schema came from a local backend or was relayed through a fal.ai WebSocket. The only difference is the transport layer.

### End-to-End Summary

```
Plugin author writes:
    Pydantic Field(ge=0.0, le=1.0, json_schema_extra=ui_field_config(order=1))
                                        |
                                        v
    Pydantic serializes to JSON schema: {"type":"number","minimum":0,"maximum":1,"ui":{...}}
                                        |
                                        v
    API returns via GET /api/v1/pipelines/schemas (local HTTP or cloud WebSocket relay)
                                        |
                                        v
    Frontend inferPrimitiveFieldType(): number + min + max = "slider"
                                        |
                                        v
    SliderField renders: <SliderWithInput min={0} max={1} step={0.01} />
                                        |
                                        v
    User moves slider -> onChange -> updateSettings(key, value)
                                        |
                                        v
    Value sent to pipeline __call__(**kwargs) on next frame
```
