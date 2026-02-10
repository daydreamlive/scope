"""Plugin system for Scope."""

from .hookspecs import hookimpl
from .manager import (
    PluginDependencyError,
    PluginInstallError,
    PluginInUseError,
    PluginManager,
    PluginNameCollisionError,
    PluginNotEditableError,
    PluginNotFoundError,
    get_plugin_manager,
    load_plugins,
    pm,
    register_plugin_input_sources,
    register_plugin_pipelines,
)

__all__ = [
    "hookimpl",
    "load_plugins",
    "pm",
    "register_plugin_input_sources",
    "register_plugin_pipelines",
    "get_plugin_manager",
    "PluginManager",
    "PluginNotFoundError",
    "PluginNotEditableError",
    "PluginInUseError",
    "PluginNameCollisionError",
    "PluginDependencyError",
    "PluginInstallError",
]
