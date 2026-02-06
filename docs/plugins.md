# Plugins

The Scope plugin system enables third-party extensions to provide custom pipelines. This guide covers how to install, manage, and develop plugins. The technical details on the plugin system can be found in the [architecture doc](./architecture/plugins.md).

## Using Plugins

### Using Plugins from the Desktop App

#### Opening Plugin Settings

1. Click the gear button in the app header to open the Settings dialog.

<img width="1382" height="86" alt="Screenshot 2026-02-02 164822" src="https://github.com/user-attachments/assets/0ca5d0ce-0fb7-4a5d-a342-d4d395db0ff8" />

2. Navigate to the Plugins tab.

<img width="657" height="399" alt="Screenshot 2026-02-02 164848" src="https://github.com/user-attachments/assets/de5d17f8-3f1d-41c1-953a-7c5b406032b8" />

#### Installing a Plugin

1. In the Plugins tab, locate the installation input field.

2. Enter a package spec (see [Plugin Sources](#plugin-sources) for format options) or browse for a local plugin directory.

3. Click the **Install** button.

4. Wait for the installation to complete and the server to restart.

<img width="414" height="112" alt="Screenshot 2026-02-02 164941" src="https://github.com/user-attachments/assets/aeac6653-7395-4a5f-a31f-d2b2e7e4730f" />

<img width="423" height="126" alt="Screenshot 2026-02-02 165047" src="https://github.com/user-attachments/assets/15ce6b29-2d9f-46d4-a3e1-86387aac8abb" />

#### Viewing Installed Plugins

The Plugins tab displays a list of all installed plugins.

#### Uninstalling a Plugin

1. Find the plugin you want to remove in the installed plugins list.

2. Click the **Uninstall** button next to the plugin.

<img width="481" height="177" alt="trashcan" src="https://github.com/user-attachments/assets/f68ac436-23d0-4b55-b4bd-d9256e551aaa" />

3. Wait for the uninstallation to complete and the server to restart.

#### Updating a Plugin

Scope automatically checks for updates when you open the Plugins tab.

1. Open the Plugins tab — any plugin with a newer version available shows an update button.

2. Click the **Update** button next to the plugin.

3. Wait for the update to complete and the server to restart.

> **Note:** Local plugins do not support update detection. To pick up code changes for a local plugin, use [Reload](#reloading-a-plugin-local-plugins-only) instead.

#### Reloading a Plugin (Local Plugins Only)

When using a local plugin directory, you can reload it after making code changes without reinstalling:

1. Find your locally installed plugin in the list.

2. Click the **Reload** button next to the plugin.

<img width="481" height="177" alt="reload" src="https://github.com/user-attachments/assets/3b22f1c8-d1a7-4fce-a322-116a2873945c" />

3. Wait for the server to restart.

### Using Plugins via Manual Installation

#### Key Differences from Desktop

The experience of using plugins with a manual installation of Scope is very similar to the experience in the desktop app with the following exceptions:

- No deep link support eg a website cannot auto-open the UI to facilitate plugin installation
- No browse button for selecting local plugin directories (you can still type a local path manually into the install field)

## Plugin Sources

Plugins can be installed from three sources:

### Git (Recommended)

Install directly from a Git repository. You can paste the URL directly from your browser:

```
https://github.com/user/plugin-repo
```

You can also specify a branch, tag, or commit:

```
https://github.com/user/plugin-repo@v1.0.0
https://github.com/user/plugin-repo@main
https://github.com/user/plugin-repo@abc1234
```

### PyPI

Install from the Python Package Index:

```
my-scope-plugin
```

You can optionally specify a version:

```
my-scope-plugin==1.0.0
```

### Local

Install from a local directory (useful for development):

```
/path/to/my-plugin
```

On Windows:

```
C:\Users\username\projects\my-plugin
```

Local plugins are installed in editable mode, meaning changes to the source code take effect after reloading the plugin.

## Developing Plugins

This section walks through creating a plugin with custom pipelines. For technical details of the plugin architecture, see the [architecture doc](./architecture/plugins.md). For technical details of the pipelines architecture, see the [architecture doc](./architecture/pipelines.md).

### Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) package manager

### Project Setup

Create a new directory for your plugin:

```
my-scope-plugin/
├── pyproject.toml
└── my_scope_plugin/
    ├── __init__.py
    ├── plugin.py
    └── pipelines/
        ├── __init__.py
        ├── schema.py
        └── pipeline.py
```

#### pyproject.toml

```toml
[project]
name = "my-scope-plugin"
version = "0.1.0"
requires-python = ">=3.12"

[project.entry-points."scope"]
my_scope_plugin = "my_scope_plugin.plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

The `[project.entry-points."scope"]` section registers your plugin with Scope. The key (`my_scope_plugin`) is your plugin name, and the value points to the module containing your hook implementation.

If your plugin needs additional third-party packages, add them to `[project.dependencies]` in `pyproject.toml` — Scope installs them automatically. You don't need to declare packages that Scope already provides (e.g. `torch`, `pydantic`) since they're available from the host environment.

#### plugin.py

```python
from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pipeline import MyPipeline

    register(MyPipeline)
```

The `register_pipelines` hook is called when Scope loads your plugin. Call `register()` for each pipeline class you want to make available.

### Creating a Text-Only Pipeline

A text-only pipeline generates video without requiring input video. This is the simplest type of pipeline.

#### Example: Color Generator

This pipeline generates solid color frames based on a configurable color.

**pipelines/schema.py:**

```python
from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults


class ColorGeneratorConfig(BasePipelineConfig):
    """Configuration for the Color Generator pipeline."""

    pipeline_id = "color-generator"
    pipeline_name = "Color Generator"
    pipeline_description = "Generates solid color frames"

    # No prompts needed
    supports_prompts = False

    # Text mode only (no video input required)
    modes = {"text": ModeDefaults(default=True)}

    # Custom parameter: the color to generate (RGB values 0-255)
    color_r: int = Field(default=128, ge=0, le=255, description="Red component")
    color_g: int = Field(default=128, ge=0, le=255, description="Green component")
    color_b: int = Field(default=128, ge=0, le=255, description="Blue component")
```

**pipelines/pipeline.py:**

```python
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline

from .schema import ColorGeneratorConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class ColorGeneratorPipeline(Pipeline):
    """Generates solid color frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ColorGeneratorConfig

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ):
        self.height = height
        self.width = width

    def __call__(self, **kwargs) -> dict:
        """Generate a solid color frame.

        Returns:
            Dict with "video" key containing tensor of shape (1, H, W, 3) in [0, 1] range.
        """
        # Read runtime parameters from kwargs (with defaults)
        color_r = kwargs.get("color_r", 128)
        color_g = kwargs.get("color_g", 128)
        color_b = kwargs.get("color_b", 128)

        # Create color tensor from current values
        color = torch.tensor([color_r / 255.0, color_g / 255.0, color_b / 255.0])

        # Create a single frame filled with our color
        frame = color.view(1, 1, 1, 3).expand(1, self.height, self.width, 3)
        return {"video": frame.clone()}
```

Key points:

- **No `prepare()` method**: Text-only pipelines don't need to request input frames
- **`modes = {"text": ModeDefaults(default=True)}`**: Declares this pipeline only supports text mode
- **`__call__` returns `{"video": tensor}`**: The tensor must be in THWC format (frames, height, width, channels) with values in [0, 1] range
- **Runtime parameters are read from `kwargs`**: Parameters like `color_r`, `color_g`, `color_b` are passed to `__call__()` and should be read using `kwargs.get()`

### Creating a Video Input Pipeline

A video input pipeline processes incoming video frames. It must implement `prepare()` to tell Scope how many input frames it needs.

#### Example: Invert Colors

This pipeline inverts the colors of input video frames.

**pipelines/schema.py:**

```python
from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults


class InvertConfig(BasePipelineConfig):
    """Configuration for the Invert Colors pipeline."""

    pipeline_id = "invert"
    pipeline_name = "Invert Colors"
    pipeline_description = "Inverts the colors of input video frames"

    # No prompts needed
    supports_prompts = False

    # Video mode only (requires video input)
    modes = {"video": ModeDefaults(default=True)}
```

**pipelines/pipeline.py:**

```python
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import InvertConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class InvertPipeline(Pipeline):
    """Inverts the colors of input video frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return InvertConfig

    def __init__(
        self,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        """Declare that we need 1 input frame."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Invert the colors of input frames.

        Args:
            video: List of input frame tensors, each (1, H, W, C) in [0, 255] range.

        Returns:
            Dict with "video" key containing inverted frames in [0, 1] range.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for InvertPipeline")

        # Stack frames into a single tensor: (T, H, W, C)
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Invert: white becomes black, black becomes white
        inverted = 1.0 - frames

        return {"video": inverted.clamp(0, 1)}
```

Key points:

- **`prepare()` returns `Requirements(input_size=N)`**: Tells Scope to collect N frames before calling `__call__`
- **`modes = {"video": ModeDefaults(default=True)}`**: Declares this pipeline only supports video mode
- **`video` parameter**: A list of tensors, one per frame, each with shape (1, H, W, C) in [0, 255] range
- **Output normalization**: Input is [0, 255], output must be [0, 1]

### Adding UI Parameters

You can expose pipeline parameters in the Scope UI by adding fields to your config with `ui_field_config()`.

Let's add an intensity slider to the Invert pipeline:

**pipelines/schema.py:**

```python
from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class InvertConfig(BasePipelineConfig):
    """Configuration for the Invert Colors pipeline."""

    pipeline_id = "invert"
    pipeline_name = "Invert Colors"
    pipeline_description = "Inverts the colors of input video frames"
    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}

    # Add a slider that appears in the Settings panel
    intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How strongly to invert the colors (0 = original, 1 = fully inverted)",
        json_schema_extra=ui_field_config(order=1, label="Intensity"),
    )
```

**pipelines/pipeline.py:**

```python
class InvertPipeline(Pipeline):
    """Inverts the colors of input video frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return InvertConfig

    def __init__(
        self,
        device: torch.device | None = None,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for InvertPipeline")

        # Read runtime parameter from kwargs (with default)
        intensity = kwargs.get("intensity", 1.0)

        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Invert with intensity blending
        inverted = 1.0 - frames
        result = frames * (1 - intensity) + inverted * intensity

        return {"video": result.clamp(0, 1)}
```

The `intensity` parameter now appears as a slider in the Settings panel when your pipeline is selected.

#### ui_field_config Options

| Option | Description |
|--------|-------------|
| `order` | Display order (lower values appear first) |
| `modes` | Restrict to specific modes, e.g., `["video"]` |
| `is_load_param` | If `True`, parameter is set when loading the pipeline and disabled during streaming |
| `label` | Short label for the UI (description becomes tooltip) |
| `category` | `"configuration"` for Settings panel (default), `"input"` for Input & Controls panel |

#### Load-time vs Runtime Parameters

Parameters are handled differently depending on `is_load_param`:

- **Load-time parameters** (`is_load_param=True`): Passed to `__init__()` when the pipeline loads. The UI control is disabled during streaming. Use for parameters that affect initialization (model selection, device, resolution).

- **Runtime parameters** (default, `is_load_param=False`): Passed to `__call__()` as kwargs on every frame. Can be adjusted during streaming. Use for parameters that affect frame processing (colors, intensity, effects).

**Important:** Runtime parameters must be read from `kwargs` in `__call__()`, not stored in `__init__()`:

```python
# Correct: Read runtime params from kwargs in __call__()
def __call__(self, **kwargs) -> dict:
    intensity = kwargs.get("intensity", 1.0)
    # Use intensity...

# Incorrect: Runtime params are NOT passed to __init__()
def __init__(self, intensity: float = 1.0, **kwargs):
    self.intensity = intensity  # This will always be the default value!
```

### Testing Your Plugin

Make code changes and then see the section on [reloading local plugins](#reloading-a-plugin-local-plugins-only).

> **Tip:** In the desktop app, you can use the browse button to select your local plugin directory. Without the desktop app, you can run `pwd` in the plugin directory to get the full path to paste into the install field which also works if you are running the server on a remote machine (eg. Runpod).
