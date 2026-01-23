import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import RIFEPipeline

    __all__ = ["RIFEPipeline"]
else:
    __all__ = []
