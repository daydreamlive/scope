"""Model downloading utilities for pose TensorRT plugin."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default HuggingFace repository for YoloNas Pose ONNX model
DEFAULT_REPO_ID = "yuvraj108c/yolo-nas-pose-onnx"
DEFAULT_ONNX_FILENAME = "yolo_nas_pose_l_0.5.onnx"  # 0.5 confidence threshold


def get_models_dir() -> Path:
    """Get the models directory for pose TensorRT models.

    Uses DAYDREAM_SCOPE_MODELS_DIR environment variable if set,
    otherwise falls back to ~/.daydream-scope/models/pose-tensorrt/

    Returns:
        Path to the models directory (created if needed)
    """
    base_dir = os.environ.get(
        "DAYDREAM_SCOPE_MODELS_DIR",
        Path.home() / ".daydream-scope" / "models",
    )
    pose_dir = Path(base_dir) / "pose-tensorrt"
    pose_dir.mkdir(parents=True, exist_ok=True)
    return pose_dir


def download_onnx_model(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_ONNX_FILENAME,
    models_dir: Path | None = None,
) -> Path:
    """Download YoloNas Pose ONNX model from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        filename: Name of the ONNX file in the repository
        models_dir: Target directory for download. Uses default if None.

    Returns:
        Path to the downloaded ONNX model
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for model downloading. "
            "Install with: pip install huggingface-hub"
        ) from e

    if models_dir is None:
        models_dir = get_models_dir()

    local_path = models_dir / filename

    if local_path.exists():
        logger.info(f"ONNX model already exists: {local_path}")
        return local_path

    logger.info(f"Downloading ONNX model from {repo_id}/{filename}...")

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
    )

    logger.info(f"Downloaded ONNX model to: {downloaded_path}")
    return Path(downloaded_path)


def get_engine_path(onnx_path: Path, gpu_name: str) -> Path:
    """Get the path for a GPU-specific TensorRT engine.

    Args:
        onnx_path: Path to the source ONNX model
        gpu_name: Sanitized GPU name for engine file naming

    Returns:
        Path where the TRT engine should be stored
    """
    # Engine stored alongside ONNX with GPU-specific suffix
    engine_name = f"{onnx_path.stem}_{gpu_name}.trt"
    return onnx_path.parent / engine_name
