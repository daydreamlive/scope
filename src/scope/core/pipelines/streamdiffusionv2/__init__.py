import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import StreamDiffusionV2Pipeline

    __all__ = ["StreamDiffusionV2Pipeline"]
else:
    __all__ = []
