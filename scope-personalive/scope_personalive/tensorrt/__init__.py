"""TensorRT acceleration for PersonaLive pipeline.

This module provides TensorRT acceleration for the PersonaLive pipeline using
NVIDIA's official TensorRT and polygraphy packages.

Installation:
    pip install daydream-scope[tensorrt]

For best performance, also install pycuda:
    pip install pycuda

Usage:
    # Convert models to TensorRT (run once)
    convert-personalive-trt --model-dir ./models --height 512 --width 512

    # The pipeline will automatically use TensorRT when engine is available
"""

# Check for TensorRT availability before importing
TRT_AVAILABLE = False
CUDA_BUFFERS_AVAILABLE = False
PYCUDA_AVAILABLE = False

try:
    import tensorrt  # noqa: F401
    from polygraphy.backend.trt import TrtRunner as _TrtRunner  # noqa: F401

    TRT_AVAILABLE = True

    # Check for zero-copy CUDA buffer support (polygraphy)
    try:
        from polygraphy.cuda import DeviceView  # noqa: F401

        CUDA_BUFFERS_AVAILABLE = True
    except ImportError:
        pass

    # Check for pycuda (best performance)
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver  # noqa: F401

        PYCUDA_AVAILABLE = True
    except ImportError:
        pass

except ImportError:
    pass

if TRT_AVAILABLE:
    from .builder import build_engine, get_engine_path, is_engine_available
    from .runner import TRTRunner

    # Import EngineModel if pycuda is available
    if PYCUDA_AVAILABLE:
        from .engine_model import EngineModel
    else:
        EngineModel = None

    __all__ = [
        "build_engine",
        "get_engine_path",
        "is_engine_available",
        "TRTRunner",
        "EngineModel",
        "TRT_AVAILABLE",
        "CUDA_BUFFERS_AVAILABLE",
        "PYCUDA_AVAILABLE",
    ]
else:
    # Provide stub functions when TRT is not available
    def get_engine_path(*args, **kwargs):
        raise RuntimeError(
            "TensorRT not available. Install with: pip install daydream-scope[tensorrt]"
        )

    def build_engine(*args, **kwargs):
        raise RuntimeError(
            "TensorRT not available. Install with: pip install daydream-scope[tensorrt]"
        )

    def is_engine_available(*args, **kwargs):
        return False

    class TRTRunner:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "TensorRT not available. Install with: pip install daydream-scope[tensorrt]"
            )

    __all__ = [
        "build_engine",
        "get_engine_path",
        "is_engine_available",
        "TRTRunner",
        "TRT_AVAILABLE",
        "CUDA_BUFFERS_AVAILABLE",
    ]
