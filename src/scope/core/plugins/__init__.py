"""Plugin system for Scope."""

from .hookspecs import hookimpl
from .manager import (
    get_event_processor,
    load_plugins,
    pm,
    register_plugin_event_processors,
    register_plugin_pipelines,
)

__all__ = [
    "get_event_processor",
    "hookimpl",
    "load_plugins",
    "pm",
    "register_plugin_event_processors",
    "register_plugin_pipelines",
]
