"""Plugin manager for discovering and loading Scope plugins."""

import logging

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
