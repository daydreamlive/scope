"""Plugin manager for discovering and loading Scope plugins."""

import logging
from pathlib import Path

import pluggy

from .hookspecs import ScopeHookSpec

logger = logging.getLogger(__name__)

# Create the plugin manager singleton
pm = pluggy.PluginManager("scope")
pm.add_hookspecs(ScopeHookSpec)


def load_plugins():
    """Discover and load all plugins via entry points."""
    pm.load_setuptools_entrypoints("scope")
    logger.info(f"Loaded {len(pm.get_plugins())} plugin(s)")


def _get_plugins_dir() -> Path:
    """Get the plugins directory path."""
    # Import here to avoid circular imports
    from scope.server.models_config import get_plugins_dir

    return get_plugins_dir()


def _parse_plugin_pyproject(pyproject_path: Path) -> dict | None:
    """Parse a plugin's pyproject.toml to extract metadata.

    Returns:
        Dict with 'pipeline_id' and 'name' if found, None otherwise
    """
    try:
        # Try tomllib (Python 3.11+) first, fall back to tomli for Python 3.10
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Get entry points to find pipeline_id
        entry_points = data.get("project", {}).get("entry-points", {}).get("scope", {})
        if entry_points:
            # Use the first entry point name as pipeline_id
            pipeline_id = next(iter(entry_points.keys()))
            return {
                "pipeline_id": pipeline_id,
                "name": data.get("project", {}).get("name", pipeline_id),
            }
        return None
    except Exception as e:
        logger.warning(f"Failed to parse {pyproject_path}: {e}")
        return None


def discover_directory_plugins() -> list[dict]:
    """Discover plugins installed in the plugins directory.

    Returns:
        List of dicts with plugin metadata (pipeline_id, name, path)
    """
    plugins = []
    plugins_dir = _get_plugins_dir()

    if not plugins_dir.exists():
        return plugins

    for plugin_dir in plugins_dir.iterdir():
        if not plugin_dir.is_dir():
            continue

        pyproject_path = plugin_dir / "pyproject.toml"
        if not pyproject_path.exists():
            continue

        metadata = _parse_plugin_pyproject(pyproject_path)
        if metadata:
            metadata["path"] = plugin_dir
            plugins.append(metadata)
            logger.debug(f"Discovered plugin: {metadata['pipeline_id']} at {plugin_dir}")

    return plugins


def register_plugin_pipelines(registry):
    """Call register_pipelines hook for all plugins.

    Args:
        registry: PipelineRegistry to register pipelines with
    """

    def register_callback(pipeline_class):
        """Callback function passed to plugins.

        For plugins, we register the config class and store the pipeline class
        module info for lazy loading by the worker.
        """
        config_class = pipeline_class.get_config_class()
        pipeline_id = config_class.pipeline_id

        # Get module path and class name for lazy loading
        module_path = pipeline_class.__module__
        class_name = pipeline_class.__name__

        registry.register(pipeline_id, config_class, module_path, class_name)
        logger.info(f"Registered plugin pipeline: {pipeline_id}")

    pm.hook.register_pipelines(register=register_callback)
