"""
Plugins configuration module for daydream-scope.

Provides centralized configuration for plugin storage location with support for:
- Default location: ~/.daydream-scope/plugins
- Environment variable override: DAYDREAM_SCOPE_PLUGINS_DIR
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default plugins directory
DEFAULT_PLUGINS_DIR = "~/.daydream-scope/plugins"

# Environment variable for overriding plugins directory
PLUGINS_DIR_ENV_VAR = "DAYDREAM_SCOPE_PLUGINS_DIR"


def get_plugins_dir() -> Path:
    """
    Get the plugins directory path.

    Priority order:
    1. DAYDREAM_SCOPE_PLUGINS_DIR environment variable
    2. Default: ~/.daydream-scope/plugins

    Returns:
        Path: Absolute path to the plugins directory
    """
    # Check environment variable first
    env_dir = os.environ.get(PLUGINS_DIR_ENV_VAR)
    if env_dir:
        plugins_dir = Path(env_dir).expanduser().resolve()
        return plugins_dir

    # Use default directory
    plugins_dir = Path(DEFAULT_PLUGINS_DIR).expanduser().resolve()
    return plugins_dir


def ensure_plugins_dir() -> Path:
    """
    Get the plugins directory path and ensure it exists.

    Returns:
        Path: Absolute path to the plugins directory
    """
    plugins_dir = get_plugins_dir()
    plugins_dir.mkdir(parents=True, exist_ok=True)
    return plugins_dir


def get_plugins_file() -> Path:
    """
    Get the path to plugins.txt file.

    This file contains the list of plugin specifiers that are installed.

    Returns:
        Path: Absolute path to plugins.txt
    """
    return get_plugins_dir() / "plugins.txt"


def get_resolved_file() -> Path:
    """
    Get the path to resolved.txt file.

    This file contains the resolved dependencies from uv pip compile.

    Returns:
        Path: Absolute path to resolved.txt
    """
    return get_plugins_dir() / "resolved.txt"
