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
        logger.info(f"Using models directory from {MODELS_DIR_ENV_VAR}: {models_dir}")
        return models_dir

    # Use default directory
    models_dir = Path(DEFAULT_MODELS_DIR).expanduser().resolve()
    logger.debug(f"Using default models directory: {models_dir}")
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


def get_required_model_files() -> list[Path]:
    """
    Get the list of required model files that should exist.

    Returns:
        list[Path]: List of required model file paths
    """
    models_dir = get_models_dir()
    return [
        models_dir / "Wan2.1-T2V-1.3B" / "config.json",
        models_dir / "WanVideo_comfy" / "umt5-xxl-enc-fp8_e4m3fn.safetensors",
        models_dir / "LongLive-1.3B" / "models" / "longlive_base.pt",
        models_dir / "StreamDiffusionV2" / "model.pt",
    ]


def models_are_downloaded() -> bool:
    """
    Check if all required model files are downloaded.

    Returns:
        bool: True if all required models are present, False otherwise
    """
    required_files = get_required_model_files()

    for file_path in required_files:
        if not file_path.exists():
            logger.info(f"Missing model file: {file_path}")
            return False

    return True
