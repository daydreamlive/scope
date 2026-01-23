"""Controller Visualizer pipeline for testing WASD and mouse input."""

import sys

from .schema import ControllerVisualizerConfig

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import ControllerVisualizerPipeline

    __all__ = ["ControllerVisualizerPipeline", "ControllerVisualizerConfig"]
else:
    __all__ = ["ControllerVisualizerConfig"]
