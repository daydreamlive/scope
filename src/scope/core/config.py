import os
from pathlib import Path

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
