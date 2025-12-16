"""Plugin system for Scope."""

from .hookspecs import hookimpl
from .manager import load_plugins, pm, register_plugin_pipelines

__all__ = ["hookimpl", "load_plugins", "pm", "register_plugin_pipelines"]
