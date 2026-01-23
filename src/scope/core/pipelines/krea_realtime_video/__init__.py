import sys

# Pipeline class requires torch which isn't available on macOS (cloud mode only)
if sys.platform != "darwin":
    from .pipeline import KreaRealtimeVideoPipeline

    __all__ = ["KreaRealtimeVideoPipeline"]
else:
    __all__ = []
