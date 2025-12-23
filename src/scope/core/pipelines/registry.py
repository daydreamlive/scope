"""Pipeline registry for centralized pipeline management.

This module provides a registry pattern to eliminate if/elif chains when
accessing pipelines by ID. It enables dynamic pipeline discovery and
metadata retrieval.

The registry separates config classes (lightweight, no heavy imports) from
pipeline classes (heavy imports, loaded lazily only when needed by worker).
"""

import importlib
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .interface import Pipeline
    from .schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Registry for managing available pipelines.

    Stores config classes (lightweight) and module paths for lazy loading
    pipeline implementations (heavy imports) only when needed.
    """

    # Map pipeline_id -> config class (lightweight, always available)
    _configs: dict[str, type["BasePipelineConfig"]] = {}

    # Map pipeline_id -> (module_path, class_name) for lazy loading
    _pipeline_modules: dict[str, tuple[str, str]] = {}

    @classmethod
    def register(
        cls,
        pipeline_id: str,
        config_class: type["BasePipelineConfig"],
        module_path: str | None = None,
        class_name: str | None = None,
    ) -> None:
        """Register a pipeline with its config class and optional module path.

        Args:
            pipeline_id: Unique identifier for the pipeline
            config_class: Pipeline config class (lightweight, from schema.py)
            module_path: Optional module path for lazy loading pipeline class
            class_name: Optional class name for lazy loading pipeline class
        """
        cls._configs[pipeline_id] = config_class
        if module_path and class_name:
            cls._pipeline_modules[pipeline_id] = (module_path, class_name)

    @classmethod
    def get(cls, pipeline_id: str) -> type["Pipeline"] | None:
        """Get a pipeline class by its ID (lazy loads the module).

        This should only be called by the worker process, not the main server.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline class if found, None otherwise
        """
        if pipeline_id not in cls._pipeline_modules:
            return None

        module_path, class_name = cls._pipeline_modules[pipeline_id]
        try:
            module = importlib.import_module(module_path, package=__package__)
            return getattr(module, class_name)
        except ImportError as e:
            logger.error(f"Failed to import pipeline {pipeline_id}: {e}")
            return None

    @classmethod
    def get_config_class(cls, pipeline_id: str) -> type["BasePipelineConfig"] | None:
        """Get config class for a specific pipeline.

        This is safe to call from the main server as it doesn't trigger
        heavy pipeline imports.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pydantic config class if found, None otherwise
        """
        return cls._configs.get(pipeline_id)

    @classmethod
    def list_pipelines(cls) -> list[str]:
        """Get list of all registered pipeline IDs.

        Returns:
            List of pipeline IDs
        """
        return list(cls._configs.keys())


def _get_gpu_vram_gb() -> float | None:
    """Get total GPU VRAM in GB if available.

    Returns:
        Total VRAM in GB if GPU is available, None otherwise
    """
    try:
        if torch.cuda.is_available():
            _, total_mem = torch.cuda.mem_get_info(0)
            return total_mem / (1024**3)
    except Exception as e:
        logger.warning(f"Failed to get GPU VRAM info: {e}")
    return None


def _should_register_pipeline(
    estimated_vram_gb: float | None, vram_gb: float | None
) -> bool:
    """Determine if a pipeline should be registered based on GPU requirements.

    Args:
        estimated_vram_gb: Estimated/required VRAM in GB from pipeline config,
            or None if no requirement
        vram_gb: Total GPU VRAM in GB, or None if no GPU

    Returns:
        True if the pipeline should be registered, False otherwise
    """
    return estimated_vram_gb is None or vram_gb is not None


# Register all available pipelines
def _register_pipelines():
    """Register pipelines based on GPU availability and requirements.

    Uses config classes from schema.py (no heavy imports) and stores
    module paths for lazy loading when worker needs them.
    """
    from .schema import (
        PIPELINE_CONFIGS,
        KreaRealtimeVideoConfig,
        LongLiveConfig,
        PassthroughConfig,
        RewardForcingConfig,
        StreamDiffusionV2Config,
    )

    # Check GPU VRAM
    vram_gb = _get_gpu_vram_gb()

    if vram_gb is not None:
        logger.info(f"GPU detected with {vram_gb:.1f} GB VRAM")
    else:
        logger.info("No GPU detected")

    # Define pipeline module paths for lazy loading
    # Maps config class -> (module_path, class_name)
    pipeline_module_paths = {
        PassthroughConfig: (".passthrough.pipeline", "PassthroughPipeline"),
        LongLiveConfig: (".longlive.pipeline", "LongLivePipeline"),
        KreaRealtimeVideoConfig: (
            ".krea_realtime_video.pipeline",
            "KreaRealtimeVideoPipeline",
        ),
        StreamDiffusionV2Config: (
            ".streamdiffusionv2.pipeline",
            "StreamDiffusionV2Pipeline",
        ),
        RewardForcingConfig: (".reward_forcing.pipeline", "RewardForcingPipeline"),
    }

    # Register each pipeline using its config class
    for pipeline_id, config_class in PIPELINE_CONFIGS.items():
        estimated_vram_gb = config_class.estimated_vram_gb

        # Check if pipeline meets GPU requirements
        should_register = _should_register_pipeline(estimated_vram_gb, vram_gb)
        if not should_register:
            logger.debug(
                f"Skipping {pipeline_id} pipeline - "
                f"does not meet GPU requirements "
                f"(required: {estimated_vram_gb} GB, "
                f"available: {vram_gb} GB)"
            )
            continue

        # Get module path for lazy loading
        module_info = pipeline_module_paths.get(config_class)
        if module_info:
            module_path, class_name = module_info
        else:
            module_path, class_name = None, None

        # Register the pipeline with its config class and module path
        PipelineRegistry.register(
            pipeline_id, config_class, module_path, class_name
        )
        logger.debug(f"Registered {pipeline_id} pipeline")


def _create_plugin_config_class(pipeline_id: str, name: str) -> type["BasePipelineConfig"]:
    """Create a minimal config class for a plugin pipeline.

    This allows plugins to appear in the pipelines list without importing
    their heavy dependencies.

    Args:
        pipeline_id: The pipeline identifier
        name: Display name for the pipeline

    Returns:
        A dynamically created config class
    """
    from .schema import BasePipelineConfig

    # Create a new class that inherits from BasePipelineConfig
    config_class = type(
        f"{name.replace('-', '_').title()}Config",
        (BasePipelineConfig,),
        {
            "pipeline_id": pipeline_id,
            "pipeline_name": name,
            "pipeline_description": f"Plugin pipeline: {name}",
            "requires_models": True,  # Assume plugins need models
        },
    )
    return config_class


def _register_directory_plugins():
    """Register plugins discovered from the plugins directory.

    These plugins are just copied to the plugins dir, not installed as packages.
    We create minimal config classes so they appear in the pipelines list.
    """
    from scope.core.plugins import discover_directory_plugins

    for plugin_info in discover_directory_plugins():
        pipeline_id = plugin_info["pipeline_id"]
        name = plugin_info.get("name", pipeline_id)

        # Skip if already registered (e.g., via entry points)
        if pipeline_id in PipelineRegistry._configs:
            logger.debug(f"Plugin {pipeline_id} already registered, skipping")
            continue

        # Create a minimal config class for the plugin
        config_class = _create_plugin_config_class(pipeline_id, name)

        # Register with no module path - the worker will discover it from the plugins dir
        PipelineRegistry.register(pipeline_id, config_class)
        logger.debug(f"Registered directory plugin: {pipeline_id}")


def _initialize_registry():
    """Initialize registry with built-in pipelines and plugins."""
    # Register built-in pipelines first
    _register_pipelines()

    # Load and register plugin pipelines via entry points (installed packages)
    from scope.core.plugins import load_plugins, register_plugin_pipelines

    load_plugins()
    register_plugin_pipelines(PipelineRegistry)

    # Also discover plugins from the plugins directory (copied plugins)
    _register_directory_plugins()

    pipeline_count = len(PipelineRegistry.list_pipelines())
    logger.info(f"Registry initialized with {pipeline_count} pipeline(s)")


# Auto-register pipelines on module import
_initialize_registry()
