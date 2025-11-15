"""
Models configuration module for daydream-scope.

Provides centralized configuration for model storage location with support for:
- Default location: ~/.daydream-scope/models
- Environment variable override: DAYDREAM_MODELS_DIR
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default models directory
DEFAULT_MODELS_DIR = "~/.daydream-scope/models"

# Environment variable for overriding models directory
MODELS_DIR_ENV_VAR = "DAYDREAM_SCOPE_MODELS_DIR"


def get_models_dir() -> Path:
    """
    Get the models directory path.

    Priority order:
    1. DAYDREAM_SCOPE_MODELS_DIR environment variable
    2. Default: ~/.daydream-scope/models

    Returns:
        Path: Absolute path to the models directory
    """
    # Check environment variable first
    env_dir = os.environ.get(MODELS_DIR_ENV_VAR)
    if env_dir:
        models_dir = Path(env_dir).expanduser().resolve()
        return models_dir

    # Use default directory
    models_dir = Path(DEFAULT_MODELS_DIR).expanduser().resolve()
    return models_dir


def ensure_models_dir() -> Path:
    """
    Get the models directory path and ensure it exists.

    Returns:
        Path: Absolute path to the models directory
    """
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_file_path(relative_path: str) -> Path:
    """
    Get the absolute path to a model file relative to the models directory.

    Args:
        relative_path: Path relative to the models directory

    Returns:
        Path: Absolute path to the model file
    """
    models_dir = get_models_dir()
    return models_dir / relative_path


def get_required_model_files(pipeline_id: str | None = None) -> list[Path]:
    """
    Get the list of required model files that should exist for a given pipeline.

    Args:
        pipeline_id: The pipeline ID to get required models for.
                     If None, returns models for all default pipelines.
                     If provided, returns models specific to that pipeline.

    Returns:
        list[Path]: List of required model file paths
    """
    models_dir = get_models_dir()

    # Passthrough doesn't need models
    if pipeline_id == "passthrough":
        return []

    # streamdiffusionv2 pipeline
    if pipeline_id == "streamdiffusionv2":
        return [
            models_dir / "Wan2.1-T2V-1.3B" / "config.json",
            models_dir / "WanVideo_comfy" / "umt5-xxl-enc-fp8_e4m3fn.safetensors",
            models_dir / "StreamDiffusionV2" / "wan_causal_dmd_v2v" / "model.pt",
        ]

    # longlive pipeline
    if pipeline_id == "longlive":
        return [
            models_dir / "Wan2.1-T2V-1.3B" / "config.json",
            models_dir / "WanVideo_comfy" / "umt5-xxl-enc-fp8_e4m3fn.safetensors",
            models_dir / "LongLive-1.3B" / "models" / "longlive_base.pt",
        ]

    # krea-realtime-video pipeline
    if pipeline_id == "krea-realtime-video":
        return [
            models_dir / "krea-realtime-video" / "krea-realtime-video-14b.safetensors",
            models_dir / "WanVideo_comfy" / "umt5-xxl-enc-fp8_e4m3fn.safetensors",
            models_dir / "Wan2.1-T2V-14B" / "config.json",
            models_dir / "Wan2.1-T2V-1.3B" / "config.json",
        ]

    # Default: nothing is required
    return []


def models_are_downloaded(pipeline_id: str | None = None) -> bool:
    """
    Check if all required model files are downloaded.

    Args:
        pipeline_id: The pipeline ID to check models for.
                     If None, checks models for all default pipelines.
                     If provided, checks models specific to that pipeline.

    Returns:
        bool: True if all required models are present, False otherwise
    """
    required_files = get_required_model_files(pipeline_id)

    for file_path in required_files:
        if not file_path.exists():
            return False

    return True
