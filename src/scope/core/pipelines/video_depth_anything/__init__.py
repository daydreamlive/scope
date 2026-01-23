import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import VideoDepthAnythingPipeline

    __all__ = ["VideoDepthAnythingPipeline"]
else:
    __all__ = []
