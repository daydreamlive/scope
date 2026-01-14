"""Plugin system for Scope."""

from .hookspecs import hookimpl
from .manager import (
    load_plugins,
    pm,
    register_plugin_pipelines,
    register_plugin_preprocessors,
)
from .preprocessor_registry import PreprocessorRegistry

__all__ = [
    "hookimpl",
    "load_plugins",
    "pm",
    "register_plugin_pipelines",
    "register_plugin_preprocessors",
    "PreprocessorRegistry",
]
