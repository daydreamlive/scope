"""Model directory utilities for optical flow TensorRT plugin."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_models_dir() -> Path:
    """Get the models directory for optical flow TensorRT models.

    Uses DAYDREAM_SCOPE_MODELS_DIR environment variable if set,
    otherwise falls back to ~/.daydream-scope/models/optical-flow-tensorrt/

    Returns:
        Path to the models directory (created if needed)
    """
    base_dir = os.environ.get(
        "DAYDREAM_SCOPE_MODELS_DIR",
        Path.home() / ".daydream-scope" / "models",
    )
    flow_dir = Path(base_dir) / "optical-flow-tensorrt"
    flow_dir.mkdir(parents=True, exist_ok=True)
    return flow_dir


def get_onnx_path(models_dir: Path, height: int, width: int) -> Path:
    """Get the path for a resolution-specific ONNX model.

    Args:
        models_dir: Base models directory
        height: Model input height
        width: Model input width

    Returns:
        Path where the ONNX model should be stored
    """
    onnx_name = f"raft_small_{height}x{width}.onnx"
    return models_dir / onnx_name


def get_engine_path(models_dir: Path, height: int, width: int, gpu_name: str) -> Path:
    """Get the path for a GPU-specific TensorRT engine.

    Args:
        models_dir: Base models directory
        height: Model input height
        width: Model input width
        gpu_name: Sanitized GPU name for engine file naming

    Returns:
        Path where the TRT engine should be stored
    """
    engine_name = f"raft_small_{height}x{width}_{gpu_name}.trt"
    return models_dir / engine_name
