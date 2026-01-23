import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import LongLivePipeline

    __all__ = ["LongLivePipeline"]
else:
    __all__ = []
