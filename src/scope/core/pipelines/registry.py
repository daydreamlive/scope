"""Pipeline registry — a filtering view over :class:`NodeRegistry`.

Pipelines and plain custom nodes share the same ``NodeRegistry._nodes``
storage. ``PipelineRegistry`` projects that storage down to entries
whose class is a :class:`Pipeline` subclass and exposes the same API
the rest of the codebase always used, so existing call sites and
plugins keep working unchanged.
"""

import importlib
import logging
from typing import TYPE_CHECKING

import torch

from scope.core.nodes.registry import NodeRegistry

if TYPE_CHECKING:
    from .interface import Pipeline
    from .schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _is_pipeline(node_class: object) -> bool:
    """Return True when ``node_class`` is a :class:`Pipeline` subclass.

    Lazily imports :class:`Pipeline` to dodge the import cycle between
    the pipelines and nodes packages at module load time.
    """
    from .interface import Pipeline

    return isinstance(node_class, type) and issubclass(node_class, Pipeline)


class PipelineRegistry:
    """Filtering view over :class:`NodeRegistry` for pipeline classes."""

    @classmethod
    def register(cls, pipeline_id: str, pipeline_class: type["Pipeline"]) -> None:
        """Plant a pipeline class into the unified :class:`NodeRegistry`.

        Delegates to ``NodeRegistry.register`` so the same logging and
        id-derivation path runs for built-in pipelines and plugin nodes
        alike. The explicit ``pipeline_id`` argument is asserted against
        the derived id to catch drift between the registry key and the
        config class's ``pipeline_id``.
        """
        config_pipeline_id = pipeline_class.get_config_class().pipeline_id
        if pipeline_id != config_pipeline_id:
            class_name = getattr(pipeline_class, "__name__", repr(pipeline_class))
            raise ValueError(
                f"Pipeline id mismatch: registered as '{pipeline_id}' but "
                f"{class_name}.get_config_class().pipeline_id is "
                f"'{config_pipeline_id}'."
            )
        NodeRegistry.register(pipeline_class)

    @classmethod
    def get(cls, pipeline_id: str) -> type["Pipeline"] | None:
        node_class = NodeRegistry.get(pipeline_id)
        return node_class if _is_pipeline(node_class) else None

    @classmethod
    def unregister(cls, pipeline_id: str) -> bool:
        if cls.get(pipeline_id) is None:
            return False
        return NodeRegistry.unregister(pipeline_id)

    @classmethod
    def is_registered(cls, pipeline_id: str) -> bool:
        return cls.get(pipeline_id) is not None

    @classmethod
    def get_config_class(cls, pipeline_id: str) -> type["BasePipelineConfig"] | None:
        pipeline_class = cls.get(pipeline_id)
        return pipeline_class.get_config_class() if pipeline_class else None

    @classmethod
    def list_pipelines(cls) -> list[str]:
        return [
            pid
            for pid in NodeRegistry.list_node_types()
            if _is_pipeline(NodeRegistry.get(pid))
        ]

    @classmethod
    def chain_produces_video(cls, pipeline_ids: list[str]) -> bool:
        """Check whether the pipeline chain produces video output.

        Returns True (the default) unless the *last* pipeline in the chain
        explicitly declares ``produces_video = False`` in its config.
        """
        if not pipeline_ids:
            return True

        # The last pipeline in the chain determines the final output modality
        last_id = pipeline_ids[-1]
        config_cls = cls.get_config_class(last_id)
        if config_cls is None:
            return True
        return getattr(config_cls, "produces_video", True)

    @classmethod
    def chain_produces_audio(cls, pipeline_ids: list[str]) -> bool:
        """Check whether any pipeline in the chain produces audio output.

        Returns True if *any* pipeline in the chain declares
        ``produces_audio = True`` in its config.  Returns False (the default)
        otherwise, so the server can skip creating an AudioProcessingTrack.
        """
        for pid in pipeline_ids:
            config_cls = cls.get_config_class(pid)
            if config_cls is not None and getattr(config_cls, "produces_audio", False):
                return True
        return False


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
    """Initialize registry with built-in pipelines, nodes, and plugins."""
    # Register built-in pipelines first
    _register_pipelines()

    # Register built-in nodes (no-op on the base abstraction branch)
    from scope.core.nodes import register_builtin_nodes

    register_builtin_nodes()

    # Load and register plugins. The unified register_plugin_nodes fires
    # both register_pipelines and register_nodes hooks, so old and new
    # plugins are picked up in one call.
    try:
        from scope.core.plugins import (
            ensure_plugins_installed,
            load_plugins,
            register_plugin_nodes,
        )

        ensure_plugins_installed()
        load_plugins()
        register_plugin_nodes()
    except Exception as e:
        logger.error(
            f"Failed to load plugins: {e}. Built-in pipelines are still available."
        )

    from scope.core.nodes.registry import NodeRegistry

    pipeline_count = len(PipelineRegistry.list_pipelines())
    node_count = len(NodeRegistry.list_node_types())
    logger.info(
        f"Registry initialized with {pipeline_count} pipeline(s) and "
        f"{node_count} node(s)"
    )


# Auto-register pipelines on module import
_initialize_registry()
