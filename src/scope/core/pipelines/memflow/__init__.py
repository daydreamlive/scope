import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import MemFlowPipeline

    __all__ = ["MemFlowPipeline"]
else:
    __all__ = []
