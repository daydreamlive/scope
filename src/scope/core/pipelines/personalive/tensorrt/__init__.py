"""TensorRT acceleration for PersonaLive pipeline.

This module provides TensorRT acceleration for the PersonaLive pipeline using
NVIDIA's official TensorRT and polygraphy packages.

Installation:
    pip install daydream-scope[tensorrt]

Usage:
    # Convert models to TensorRT (run once)
    convert-personalive-trt --model-dir ./models --height 512 --width 512

    # The pipeline will automatically use TensorRT when engine is available
"""

# Check for TensorRT availability before importing
TRT_AVAILABLE = False
try:
    import tensorrt  # noqa: F401
    from polygraphy.backend.trt import TrtRunner as _TrtRunner  # noqa: F401

    TRT_AVAILABLE = True
except ImportError:
    pass

if TRT_AVAILABLE:
    from .builder import build_engine, get_engine_path, is_engine_available
    from .runner import TRTRunner

    __all__ = [
        "build_engine",
        "get_engine_path",
        "is_engine_available",
        "TRTRunner",
        "TRT_AVAILABLE",
    ]
else:
    # Provide stub functions when TRT is not available
    def get_engine_path(*args, **kwargs):
        raise RuntimeError("TensorRT not available. Install with: pip install daydream-scope[tensorrt]")

    def build_engine(*args, **kwargs):
        raise RuntimeError("TensorRT not available. Install with: pip install daydream-scope[tensorrt]")

    def is_engine_available(*args, **kwargs):
        return False

    class TRTRunner:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorRT not available. Install with: pip install daydream-scope[tensorrt]")

    __all__ = [
        "build_engine",
        "get_engine_path",
        "is_engine_available",
        "TRTRunner",
        "TRT_AVAILABLE",
    ]
