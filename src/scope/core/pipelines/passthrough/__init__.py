import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import PassthroughPipeline

    __all__ = ["PassthroughPipeline"]
else:
    __all__ = []
