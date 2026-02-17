"""Pipeline registry for centralized pipeline management.

This module provides a registry pattern to eliminate if/elif chains when
accessing pipelines by ID. It enables dynamic pipeline discovery and
metadata retrieval.
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
    """Registry for managing available pipelines."""

    _pipelines: dict[str, type["Pipeline"]] = {}

    @classmethod
    def register(cls, pipeline_id: str, pipeline_class: type["Pipeline"]) -> None:
        """Register a pipeline class with its ID.

        Args:
            pipeline_id: Unique identifier for the pipeline
            pipeline_class: Pipeline class to register
        """
        cls._pipelines[pipeline_id] = pipeline_class

    @classmethod
    def get(cls, pipeline_id: str) -> type["Pipeline"] | None:
        """Get a pipeline class by its ID.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline class if found, None otherwise
        """
        return cls._pipelines.get(pipeline_id)

    @classmethod
    def unregister(cls, pipeline_id: str) -> bool:
        """Remove a pipeline from the registry.

        Args:
            pipeline_id: Pipeline identifier to remove

        Returns:
            True if pipeline was removed, False if not found
        """
        if pipeline_id in cls._pipelines:
            del cls._pipelines[pipeline_id]
            return True
        return False

    @classmethod
    def is_registered(cls, pipeline_id: str) -> bool:
        """Check if a pipeline is registered.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if pipeline is registered, False otherwise
        """
        return pipeline_id in cls._pipelines

    @classmethod
    def get_config_class(cls, pipeline_id: str) -> type["BasePipelineConfig"] | None:
        """Get config class for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pydantic config class if found, None otherwise
        """
        pipeline_class = cls.get(pipeline_id)
        if pipeline_class is None:
            return None
        return pipeline_class.get_config_class()

    @classmethod
    def list_pipelines(cls) -> list[str]:
        """Get list of all registered pipeline IDs.

        Returns:
            List of pipeline IDs
        """
        return list(cls._pipelines.keys())


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
    """Register pipelines based on GPU availability and requirements."""
    # Check GPU VRAM
    vram_gb = _get_gpu_vram_gb()

    if vram_gb is not None:
        logger.info(f"GPU detected with {vram_gb:.1f} GB VRAM")
    else:
        logger.info("No GPU detected")

    # Define pipeline imports with their module paths and class names
    pipeline_configs = [
        (
            "streamdiffusionv2",
            ".streamdiffusionv2.pipeline",
            "StreamDiffusionV2Pipeline",
        ),
        ("longlive", ".longlive.pipeline", "LongLivePipeline"),
        (
            "krea_realtime_video",
            ".krea_realtime_video.pipeline",
            "KreaRealtimeVideoPipeline",
        ),
        (
            "reward_forcing",
            ".reward_forcing.pipeline",
            "RewardForcingPipeline",
        ),
        ("memflow", ".memflow.pipeline", "MemFlowPipeline"),
        ("passthrough", ".passthrough.pipeline", "PassthroughPipeline"),
        (
            "video_depth_anything",
            ".video_depth_anything.pipeline",
            "VideoDepthAnythingPipeline",
        ),
        (
            "controller-viz",
            ".controller_viz.pipeline",
            "ControllerVisualizerPipeline",
        ),
        ("rife", ".rife.pipeline", "RIFEPipeline"),
        ("scribble", ".scribble.pipeline", "ScribblePipeline"),
        ("gray", ".gray.pipeline", "GrayPipeline"),
        ("optical_flow", ".optical_flow.pipeline", "OpticalFlowPipeline"),
    ]

    # Try to import and register each pipeline
    for pipeline_name, module_path, class_name in pipeline_configs:
        # Try to import the pipeline first to get its config
        try:
            module = importlib.import_module(module_path, package=__package__)
            pipeline_class = getattr(module, class_name)

            # Get the config class to check VRAM requirements
            config_class = pipeline_class.get_config_class()
            estimated_vram_gb = config_class.estimated_vram_gb

            # Check if pipeline meets GPU requirements
            should_register = _should_register_pipeline(estimated_vram_gb, vram_gb)
            if not should_register:
                logger.debug(
                    f"Skipping {pipeline_name} pipeline - "
                    f"does not meet GPU requirements "
                    f"(required: {estimated_vram_gb} GB, "
                    f"available: {vram_gb} GB)"
                )
                continue

            # Register the pipeline
            PipelineRegistry.register(config_class.pipeline_id, pipeline_class)
            logger.debug(
                f"Registered {pipeline_name} pipeline (ID: {config_class.pipeline_id})"
            )
        except ImportError as e:
            logger.warning(
                f"Could not import {pipeline_name} pipeline: {e}. "
                f"This pipeline will not be available."
            )
        except Exception as e:
            logger.warning(
                f"Error loading {pipeline_name} pipeline: {e}. "
                f"This pipeline will not be available."
            )


def _initialize_registry():
    """Initialize registry with built-in pipelines and plugins."""
    # Register built-in pipelines first
    _register_pipelines()

    # Load and register plugin pipelines
    try:
        from scope.core.plugins import (
            ensure_plugins_installed,
            load_plugins,
            register_plugin_pipelines,
        )

        ensure_plugins_installed()
        load_plugins()
        register_plugin_pipelines(PipelineRegistry)
    except Exception as e:
        logger.error(
            f"Failed to load plugins: {e}. Built-in pipelines are still available."
        )

    pipeline_count = len(PipelineRegistry.list_pipelines())
    logger.info(f"Registry initialized with {pipeline_count} pipeline(s)")


# Auto-register pipelines on module import
_initialize_registry()
