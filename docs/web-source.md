# Using a Web App as a Video Source

Scope doesn't include a built-in browser capture input source, but there are two practical paths for getting web content (HTML canvas, WebGL animations, single-page apps) into a pipeline: routing through a video-sharing tool like OBS, or building a lightweight plugin that captures frames directly from a URL.

---

## Option 1: Route through OBS (recommended for live use)

[OBS Studio](https://obsproject.com/) has a built-in **Browser Source** that renders any URL at a configurable resolution. You can forward that output into Scope via Syphon (macOS), Spout (Windows), or NDI (any platform).

This approach gives you real-time GPU-accelerated capture without writing any code, and works with any web app that runs in a Chromium-based browser.

### macOS: OBS → Syphon

<details>
<summary>Requirements: macOS 11+, OBS 30+, OBS Syphon plugin</summary>

Install the [obs-syphon plugin](https://github.com/zakk4223/SyphonInject) and restart OBS before continuing.
</details>

1. Open OBS and add a **Browser Source** in your scene. Set the URL to your web app (e.g. `http://localhost:3000`) and configure the width/height to match your target resolution.
2. In OBS, go to **Tools → Syphon** and enable output for your scene.
3. In Scope, select **Video** for Input Mode under **Input & Controls**.
4. Select **Syphon** for **Video Source** and choose the OBS server from the dropdown.

See the [Syphon guide](./syphon.md) for full Scope Syphon receiver setup.

### Windows: OBS → Spout

OBS on Windows includes Spout output support via the [obs-spout2 plugin](https://github.com/Off-World-Live/obs-spout2-plugin).

1. Add a **Browser Source** in OBS pointed at your web app URL.
2. In OBS, add a **Spout2 Output** filter to the scene or source.
3. In Scope, select **Video** for Input Mode and **Spout Receiver** for **Video Source**.

See the [Spout guide](./spout.md) for full Scope Spout receiver setup.

### Cross-platform: OBS → NDI

1. Install the [obs-ndi plugin](https://github.com/obs-ndi/obs-ndi) for OBS.
2. Add a **Browser Source** in OBS and enable NDI output in **Tools → NDI Output Settings**.
3. In Scope, select **Video** for Input Mode and **NDI** for **Video Source**, then choose the OBS source from the dropdown.

See the [NDI guide](./ndi.md) for full Scope NDI receiver setup.

---

## Option 2: Write a web capture plugin

If you want a self-contained solution without OBS, you can write a Scope plugin that uses [Playwright](https://playwright.dev/python/) to render a URL headlessly and capture frames. The plugin registers as a text-mode pipeline, meaning it acts as a source rather than a processor — no video input is needed.

> **Note:** Screenshot-based capture typically achieves 5 to 15 fps depending on page complexity, which is lower than real-time GPU sharing via Syphon or Spout. This approach is best suited for relatively static content or when real-time performance is not required.

### Project structure

```
scope-web-capture/
├── pyproject.toml
└── web_capture/
    ├── __init__.py
    ├── plugin.py
    └── pipelines/
        ├── __init__.py
        ├── schema.py
        └── pipeline.py
```

### pyproject.toml

```toml
[project]
name = "scope-web-capture"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["playwright>=1.40.0", "pillow"]

[project.entry-points."scope"]
web_capture = "web_capture.plugin"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### plugin.py

```python
from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    from .pipelines.pipeline import WebCapturePipeline

    register(WebCapturePipeline)
```

### pipelines/schema.py

```python
from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, ui_field_config


class WebCaptureConfig(BasePipelineConfig):
    """Configuration for the Web Capture pipeline."""

    pipeline_id = "web-capture"
    pipeline_name = "Web Capture"
    pipeline_description = "Captures frames from a web page URL (HTML, canvas, WebGL)"
    supports_prompts = False
    modes = {"text": ModeDefaults(default=True)}

    url: str = Field(
        default="http://localhost:3000",
        description="URL of the web app to capture",
        json_schema_extra=ui_field_config(order=1, label="URL", is_load_param=True),
    )
    width: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Capture width in pixels",
        json_schema_extra=ui_field_config(order=2, label="Width", is_load_param=True),
    )
    height: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Capture height in pixels",
        json_schema_extra=ui_field_config(order=3, label="Height", is_load_param=True),
    )
```

### pipelines/pipeline.py

```python
import io
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.pipelines.interface import Pipeline

from .schema import WebCaptureConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class WebCapturePipeline(Pipeline):
    """Captures frames from a web page as a video source."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return WebCaptureConfig

    def __init__(
        self,
        url: str = "http://localhost:3000",
        width: int = 512,
        height: int = 512,
        **kwargs,
    ):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(viewport={"width": width, "height": height})
        self._page.goto(url)
        self._width = width
        self._height = height

    def __call__(self, **kwargs) -> dict:
        """Capture a frame from the web page.

        Returns:
            Dict with "video" key containing a tensor of shape (1, H, W, 3) in [0, 1] range.
        """
        from PIL import Image

        screenshot_bytes = self._page.screenshot(type="png")
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, 3)
        return {"video": tensor}

    def close(self):
        try:
            self._page.close()
            self._browser.close()
            self._playwright.stop()
        except Exception:
            pass
```

Key points:

- **`modes = {"text": ModeDefaults(default=True)}`**: This pipeline generates video without requiring input frames, just like the Color Generator example in the [plugin development guide](./plugins.md).
- **`is_load_param=True`**: The URL, width, and height are set when the pipeline loads and cannot be changed while streaming. Changing them requires reloading.
- **`close()`**: Overriding `close()` ensures the headless browser is cleaned up when the pipeline is unloaded.

### Installing and running

1. Install the plugin from the local directory:

   ```bash
   uv run daydream-scope install -e /path/to/scope-web-capture
   ```

   Or use the **Install** button in **Settings → Plugins** and browse to the directory.

2. After installation, install the Playwright browser binary. Open a terminal in your Scope environment and run:

   ```bash
   uv run playwright install chromium
   ```

3. Reload or restart Scope. The **Web Capture** node will appear in the pipeline list.

4. Set the **URL** to your web app (e.g. `http://localhost:3000`) and load the pipeline.

---

## Choosing an approach

| Approach | Platform | Frame rate | Setup effort |
| -------- | -------- | ---------- | ------------ |
| OBS → Syphon | macOS only | Real-time | Medium (OBS + plugin) |
| OBS → Spout | Windows only | Real-time | Medium (OBS + plugin) |
| OBS → NDI | Any | Real-time | Medium (OBS + plugin) |
| Web Capture plugin | Any | 5 to 15 fps | Low (code-based) |

If you are building for live performance and need real-time frame rates, the OBS path is the better choice. If you are prototyping or your web app is relatively static (particle systems at low update rates, generative art, data visualizations), the plugin approach is simpler to set up and keeps everything self-contained within Scope.

---

## See also

- [Using Syphon](./syphon.md) — near-zero latency GPU texture sharing on macOS
- [Using Spout](./spout.md) — GPU texture sharing on Windows
- [Using NDI](./ndi.md) — network video sharing across machines
- [Developing Plugins](./plugins.md) — full guide to building custom pipeline plugins
