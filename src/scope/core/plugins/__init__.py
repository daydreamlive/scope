"""Plugin system for Scope."""

from .hookspecs import hookimpl
from .manager import (
    FailedPluginInfo,
    PluginDependencyError,
    PluginInstallError,
    PluginInUseError,
    PluginManager,
    PluginNameCollisionError,
    PluginNotEditableError,
    PluginNotFoundError,
    ensure_plugins_installed,
    get_plugin_manager,
    load_plugins,
    pm,
    register_plugin_pipelines,
)

__all__ = [
    "hookimpl",
    "ensure_plugins_installed",
    "load_plugins",
    "pm",
    "register_plugin_pipelines",
    "get_plugin_manager",
    "FailedPluginInfo",
    "PluginManager",
    "PluginNotFoundError",
    "PluginNotEditableError",
    "PluginInUseError",
    "PluginNameCollisionError",
    "PluginDependencyError",
    "PluginInstallError",
]
