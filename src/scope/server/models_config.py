"""
Models configuration module for daydream-scope.

Provides centralized configuration for model storage location with support for:
- Default location: ~/.daydream-scope/models
- Environment variable override: DAYDREAM_SCOPE_MODELS_DIR
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
    Also ensures the models/lora subdirectory exists.

    Returns:
        Path: Absolute path to the models directory
    """
    models_dir = get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the lora subdirectory exists
    lora_dir = models_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)

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


def get_assets_dir() -> Path:
    """
    Get the assets directory path (at the same level as models directory).

    If DAYDREAM_SCOPE_MODELS_DIR is set, assets directory will be at the same level.
    Otherwise, defaults to ~/.daydream-scope/assets

    Returns:
        Path: Absolute path to the assets directory
    """
    models_dir = get_models_dir()
    # Get the parent directory (e.g., ~/.daydream-scope) and create assets directory there
    assets_dir = models_dir.parent / "assets"
    return assets_dir


def get_required_model_files(pipeline_id: str | None = None) -> list[Path]:
    """
    Get the list of required model files that should exist for a given pipeline.

    Args:
        pipeline_id: The pipeline ID to get required models for.

    Returns:
        list[Path]: List of required model file paths
    """
    models_dir = get_models_dir()

    from scope.core.pipelines.artifacts import (
        GoogleDriveArtifact,
        HuggingfaceRepoArtifact,
    )

    from .artifact_registry import get_artifacts_for_pipeline

    if pipeline_id == "passthrough" or pipeline_id is None:
        return []

    artifacts = get_artifacts_for_pipeline(pipeline_id)
    if not artifacts:
        return []

    required_files = []
    for artifact in artifacts:
        if isinstance(artifact, HuggingfaceRepoArtifact):
            local_dir_name = artifact.repo_id.split("/")[-1]
            # Add each file from the artifact's files list
            for file in artifact.files:
                required_files.append(models_dir / local_dir_name / file)
        elif isinstance(artifact, GoogleDriveArtifact):
            # For Google Drive artifacts, use name if specified, otherwise use models_dir
            if artifact.name:
                output_dir = models_dir / artifact.name
            else:
                output_dir = models_dir

            # If files are specified, add all files from the artifact
            if artifact.files:
                for filename in artifact.files:
                    required_files.append(output_dir / filename)
            else:
                # If files not specified, check for file_id as filename
                required_files.append(output_dir / artifact.file_id)
        else:
            logger.warning(f"Unknown artifact type: {type(artifact)}")

    return required_files


def models_are_downloaded(pipeline_id: str) -> bool:
    """
    Check if all required model files are downloaded and non-empty.

    Args:
        pipeline_id: The pipeline ID to check models for.

    Returns:
        bool: True if all required models are present and non-empty, False otherwise
    """
    required_files = get_required_model_files(pipeline_id)

    for file_path in required_files:
        # Check if path exists
        if not file_path.exists():
            return False

        # If it's a file, check it's non-empty
        if file_path.is_file():
            if file_path.stat().st_size == 0:
                return False

        # If it's a directory, check it's non-empty
        elif file_path.is_dir():
            if not any(file_path.iterdir()):
                return False

    return True
