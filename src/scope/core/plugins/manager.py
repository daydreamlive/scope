"""Plugin manager for discovering and loading Scope plugins."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pluggy

from .hookspecs import ScopeHookSpec

if TYPE_CHECKING:
    from scope.core.events import EventProcessor

logger = logging.getLogger(__name__)

# Create the plugin manager singleton
pm = pluggy.PluginManager("scope")
pm.add_hookspecs(ScopeHookSpec)

# Event processor registry
_event_processors: dict[str, EventProcessor] = {}
_event_processors_loaded = False


def load_plugins():
    """Discover and load all plugins via entry points."""
    pm.load_setuptools_entrypoints("scope")
    logger.info(f"Loaded {len(pm.get_plugins())} plugin(s)")


def register_plugin_pipelines(registry):
    """Call register_pipelines hook for all plugins.

    Args:
        registry: PipelineRegistry to register pipelines with
    """

    def register_callback(pipeline_class):
        """Callback function passed to plugins."""
        config_class = pipeline_class.get_config_class()
        pipeline_id = config_class.pipeline_id
        registry.register(pipeline_id, pipeline_class)
        logger.info(f"Registered plugin pipeline: {pipeline_id}")

    pm.hook.register_pipelines(register=register_callback)


def register_plugin_event_processors():
    """Call register_event_processors hook for all plugins.

    This populates the event processor registry with processors from plugins.
    """
    global _event_processors_loaded
    if _event_processors_loaded:
        return

    def register_callback(name: str, processor: EventProcessor):
        """Callback function passed to plugins."""
        _event_processors[name] = processor
        logger.info(f"Registered event processor: {name}")

    pm.hook.register_event_processors(register=register_callback)
    _event_processors_loaded = True


def get_event_processor(name: str) -> EventProcessor | None:
    """Get a registered event processor by name.

    Args:
        name: The processor name (e.g., "prompt_enhancer", "image_generator")

    Returns:
        The registered EventProcessor, or None if not found.
    """
    # Ensure event processors are loaded
    if not _event_processors_loaded:
        register_plugin_event_processors()

    return _event_processors.get(name)
