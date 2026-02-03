# Pipeline Architecture

## Overview

Pipelines are the core abstraction for handling streaming video in Scope. A pipeline encapsulates:

- Model loading and inference logic
- Configuration schema defining available parameters
- Metadata (name, description, VRAM requirements, feature flags)
- Input/output modes (text-to-video, video-to-video)

The pipeline system enables a plugin-based architecture where third-party pipelines can be used as well which is described in the [architecture doc](plugins.md).

## Pipeline Definition

### The Pipeline Base Class

All pipelines inherit from the abstract `Pipeline` class.

```python
from abc import ABC, abstractmethod

class Pipeline(ABC):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the Pydantic config class for this pipeline."""
        ...

    @abstractmethod
    def __call__(self, **kwargs) -> dict:
        """Process video frames and return results."""
        pass
```

**Key methods:**

| Method | Purpose |
|--------|---------|
| `get_config_class()` | Returns the Pydantic config class that defines parameters and metadata |
| `__call__()` | Processes input frames and returns generated video |

### Configuration Schema

Every pipeline defines a Pydantic configuration class that inherits from `BasePipelineConfig`. This class serves as the **single source of truth** for:

1. Pipeline metadata (ID, name, description, version)
2. Feature flags (LoRA support, VACE support, quantization)
3. Parameter definitions with validation constraints
4. UI rendering hints for the frontend

Example from `longlive/schema.py`:

```python
from pydantic import Field
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config

class LongLiveConfig(BasePipelineConfig):
    # Metadata (class variables)
    pipeline_id = "longlive"
    pipeline_name = "LongLive"
    pipeline_description = "A streaming autoregressive video diffusion model..."
    estimated_vram_gb = 20.0
    supports_lora = True
    supports_vace = True

    # Parameters (instance fields with validation)
    height: int = Field(
        default=320,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )

    noise_scale: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Amount of noise to add during video generation",
        json_schema_extra=ui_field_config(
            order=7, component="noise", modes=["video"], is_load_param=True
        ),
    )
```

### Pipeline Metadata

Configuration classes declare metadata as class variables:

| Metadata | Type | Description |
|----------|------|-------------|
| `pipeline_id` | `str` | Unique identifier used for registry lookup |
| `pipeline_name` | `str` | Human-readable display name |
| `pipeline_description` | `str` | Description of capabilities |
| `pipeline_version` | `str` | Semantic version string |
| `docs_url` | `str \| None` | Link to pipeline documentation |
| `estimated_vram_gb` | `float \| None` | Estimated VRAM requirement in GB |
| `artifacts` | `list[Artifact]` | Model files required by the pipeline |

**Feature flags** control which UI controls are shown:

| Flag | Effect |
|------|--------|
| `supports_lora` | Enables LoRA management UI |
| `supports_vace` | Enables VACE reference image UI |
| `supports_cache_management` | Enables cache controls |
| `supports_quantization` | Enables quantization selector |
| `supports_kv_cache_bias` | Enables KV cache bias slider |

### Artifacts

Artifacts declare model files and resources that a pipeline requires. The system downloads these automatically before the pipeline loads.

**Available artifact types:**

| Type | Source | Attributes |
|------|--------|------------|
| `HuggingfaceRepoArtifact` | HuggingFace Hub | `repo_id`, `files` |
| `GoogleDriveArtifact` | Google Drive | `file_id`, `files` (optional), `name` (optional) |

**Example usage:**

```python
from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact, GoogleDriveArtifact

class MyPipelineConfig(BasePipelineConfig):
    pipeline_id = "my-pipeline"
    # ...

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="depth-anything/Video-Depth-Anything-Small",
            files=["video_depth_anything_vits.pth"],
        ),
        GoogleDriveArtifact(
            file_id="1Smy6gY7BkS_RzCjPCbMEy-TsX8Ma5B0R",
            files=["flownet.pkl"],
            name="RIFE",
        ),
    ]
```

Common artifacts shared across pipelines are defined in `common_artifacts.py` and can be reused:

```python
from scope.core.pipelines.common_artifacts import WAN_1_3B_ARTIFACT, VACE_ARTIFACT

class MyPipelineConfig(BasePipelineConfig):
    artifacts = [WAN_1_3B_ARTIFACT, VACE_ARTIFACT]
```

### Input Requirements

The `prepare()` method declares input frame requirements before processing. **Pipelines that accept video input must implement it** — the frame processor calls `prepare()` to determine how many frames to buffer before calling `__call__()`. Text-only pipelines (no video input) can return `None` or omit the method entirely.

**Purpose:** The frame processor calls `prepare()` before each generation cycle to determine how many input frames to collect before calling `__call__()`.

**Return type:** `Requirements(input_size=N)` or `None`

| Return Value | Meaning |
|--------------|---------|
| `Requirements(input_size=N)` | Pipeline needs `N` input frames before `__call__()` |
| `None` | Pipeline operates in text-only mode (no video input needed) |

**Simple pipeline example** (fixed input size):

```python
from scope.core.pipelines.interface import Requirements

class PassthroughPipeline(Pipeline):
    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")  # List of 4 frames
        # ... process video ...
```

**Multi-mode pipeline example** (text and video modes):

Pipelines that support both text-to-video and video-to-video modes use the `prepare_for_mode()` helper from `defaults.py`:

```python
from scope.core.pipelines.defaults import prepare_for_mode
from scope.core.pipelines.interface import Requirements

class LongLivePipeline(Pipeline):
    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode."""
        return prepare_for_mode(self.__class__, self.components.config, kwargs)
```

The helper returns `Requirements` when video mode is active (indicated by `video=True` in kwargs) and `None` for text mode. The `input_size` is calculated from the pipeline's configuration.

**Why implement `prepare()`:**

- Without it, the frame processor cannot know how many frames to buffer before calling `__call__()`
- Enables efficient queue management—the processor sizes queues based on requirements
- Allows multi-mode pipelines to dynamically switch between text and video input modes

### Mode System

Pipelines can support multiple input modes with different default parameters:

```python
class LongLiveConfig(BasePipelineConfig):
    # Base defaults
    height: int = Field(default=320, ...)
    width: int = Field(default=576, ...)

    # Mode-specific overrides
    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(
            height=512,
            width=512,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
        ),
    }
```

**Available modes:**

| Mode | Description |
|------|-------------|
| `text` | Text-to-video generation from prompts only |
| `video` | Video-to-video with input conditioning |

The `default=True` flag marks which mode is selected initially. Mode-specific defaults override base defaults when the user switches modes.

---

## Configuration Schemas

### JSON Schema Generation

Pydantic models automatically generate JSON Schema via `model_json_schema()`. The backend exposes this schema through the `/pipelines` endpoint, which the frontend consumes for dynamic UI rendering.

The schema includes:
- Field types and validation constraints (`minimum`, `maximum`, `enum`)
- Default values
- Descriptions (used as tooltips)
- Custom UI metadata (via `json_schema_extra`)

### UI Metadata

The `ui_field_config()` helper attaches rendering hints to schema fields:

```python
from ..base_schema import ui_field_config

height: int = Field(
    default=320,
    ge=1,
    description="Output height in pixels",
    json_schema_extra=ui_field_config(
        order=4,              # Display order (lower = higher)
        component="resolution", # Complex component grouping
        is_load_param=True,   # Requires pipeline reload
        modes=["video"],      # Only show in video mode
        label="Height",       # Short label (description used for tooltip)
        category="configuration", # Panel placement
    ),
)
```

**UI metadata fields:**

| Field | Type | Description |
|-------|------|-------------|
| `order` | `int` | Sort order for display (lower values appear first) |
| `component` | `str` | Groups fields into complex widgets (e.g., "resolution", "noise") |
| `modes` | `list[str]` | Restrict visibility to specific input modes |
| `is_load_param` | `bool` | If `True`, field is disabled during streaming (requires reload) |
| `label` | `str` | Short display label; description becomes tooltip |
| `category` | `str` | `"configuration"` for Settings panel, `"input"` for Input & Controls |

### Load-time vs Runtime Parameters

Parameters are categorized by when they can be changed:

| Type | `is_load_param` | Editable During Streaming | Example |
|------|-----------------|---------------------------|---------|
| Load parameter | `True` | No | Resolution, quantization, seed |
| Runtime parameter | `False` | Yes | Prompt strength, noise scale |

Load parameters are passed when the pipeline is loaded and require a restart to change. Runtime parameters can be adjusted while streaming.

---

## Dynamic UI Rendering

### Schema-to-UI Flow

```
Backend (Python)                    Frontend (React)
─────────────────                   ─────────────────
BasePipelineConfig
       │
       ▼
model_json_schema()
       │
       ▼
GET /pipelines  ───────────────────▶  configSchema
                                           │
                                           ▼
                                    parseConfigurationFields()
                                           │
                                           ▼
                                    Field type inference
                                           │
                                    ┌──────┴──────┐
                                    ▼             ▼
                              Primitive      Complex
                              widgets        components
```

### Field Type Inference

For fields without a `component` specified, the frontend automatically renders an appropriate widget based on the JSON Schema type. The frontend (`schemaSettings.ts`) infers widget types from schema properties:

| Schema Property | Inferred Type | Widget |
|-----------------|---------------|--------|
| `type: "boolean"` | `toggle` | Toggle switch |
| `type: "string"` | `text` | Text input |
| `type: "number"` with `minimum`/`maximum` | `slider` | Slider with input |
| `type: "number"` without bounds | `number` | Number input |
| `enum` or `$ref` | `enum` | Select dropdown |

### Two-Tier Component System

**Primitive fields** render as individual widgets based on inferred type.

**Complex components** group related fields into unified UI blocks:

| Component | Fields | Rendered As |
|-----------|--------|-------------|
| `resolution` | `height`, `width` | Linked dimension inputs |
| `noise` | `noise_scale`, `noise_controller` | Slider + toggle |
| `vace` | `vace_context_scale` | VACE configuration panel |
| `lora` | `lora_merge_strategy` | LoRA manager |
| `denoising_steps` | `denoising_steps` | Multi-step slider |
| `cache` | `manage_cache` | Cache controls |
| `quantization` | `quantization` | Quantization selector |
| `image` | Image path fields | Image picker |

Fields with the same `component` value are grouped and rendered once.

### Mode-Aware Filtering

Fields specify which modes they appear in via `modes`:

```python
noise_scale: float = Field(
    ...,
    json_schema_extra=ui_field_config(modes=["video"]),
)
```

The frontend filters fields based on the current input mode, hiding irrelevant controls.

---

## Pipeline Registry

### Registration

Pipelines register with the central `PipelineRegistry` at startup:

```python
from scope.core.pipelines.registry import PipelineRegistry

PipelineRegistry.register(config_class.pipeline_id, pipeline_class)
```

The registry maintains a mapping of pipeline IDs to their implementation classes.

### Built-in vs Plugin Pipelines

**Built-in pipelines** are registered automatically when the `registry` module is imported. The `_register_pipelines()` function loads each built-in pipeline and registers it if GPU requirements are met.

**Plugin pipelines** register through the pluggy hook system:

```python
from scope.core.pipelines.hookspecs import hookimpl

@hookimpl
def register_pipelines(register):
    from .pipelines import MyCustomPipeline
    register(MyCustomPipeline)
```

See [plugins.md](plugins.md) for details on plugin development and installation.

### GPU-Based Filtering

Pipelines with `estimated_vram_gb` set are only registered if a compatible GPU is detected. This prevents showing pipelines that cannot run on the current hardware.

---

## Creating a Pipeline

### Minimal Example

```python
# my_pipeline/schema.py
from pydantic import Field
from scope.core.pipelines.base_schema import BasePipelineConfig, ui_field_config

class MyPipelineConfig(BasePipelineConfig):
    pipeline_id = "my-pipeline"
    pipeline_name = "My Pipeline"
    pipeline_description = "A custom video generation pipeline"

    strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Generation strength",
        json_schema_extra=ui_field_config(order=1),
    )


# my_pipeline/pipeline.py
from scope.core.pipelines.interface import Pipeline, Requirements
from .schema import MyPipelineConfig

class MyPipeline(Pipeline):
    @classmethod
    def get_config_class(cls):
        return MyPipelineConfig

    def prepare(self, **kwargs) -> Requirements:
        """Declare how many input frames are needed."""
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")  # List of 4 tensors, each (1, H, W, C)
        # ... process video ...
        return {"video": output_tensor}
```
