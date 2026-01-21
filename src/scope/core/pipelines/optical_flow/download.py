"""Model directory utilities for optical flow pipeline."""

import logging
from pathlib import Path

from scope.core.config import get_model_file_path

logger = logging.getLogger(__name__)


def get_models_dir() -> Path:
    """Get the models directory for optical flow models.

    Returns:
        Path to the models directory (created if needed)
    """
    flow_dir = get_model_file_path("optical-flow")
    flow_dir.mkdir(parents=True, exist_ok=True)
    return flow_dir


def get_onnx_path(
    models_dir: Path, height: int, width: int, model_name: str = "raft_small"
) -> Path:
    """Get the path for a resolution-specific ONNX model.

    Args:
        models_dir: Base models directory
        height: Model input height
        width: Model input width
        model_name: Model name ("raft_small" or "raft_large")

    Returns:
        Path where the ONNX model should be stored
    """
    onnx_name = f"{model_name}_{height}x{width}.onnx"
    return models_dir / onnx_name


def get_engine_path(
    models_dir: Path,
    height: int,
    width: int,
    gpu_name: str,
    model_name: str = "raft_small",
) -> Path:
    """Get the path for a GPU-specific TensorRT engine.

    Args:
        models_dir: Base models directory
        height: Model input height
        width: Model input width
        gpu_name: Sanitized GPU name for engine file naming
        model_name: Model name ("raft_small" or "raft_large")

    Returns:
        Path where the TRT engine should be stored
    """
    engine_name = f"{model_name}_{height}x{width}_{gpu_name}.trt"
    return models_dir / engine_name
