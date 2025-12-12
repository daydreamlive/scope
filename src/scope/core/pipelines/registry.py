"""Pipeline registry for centralized pipeline management.

This module provides a registry pattern to eliminate if/elif chains when
accessing pipelines by ID. It enables dynamic pipeline discovery and
metadata retrieval.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import Pipeline
    from .schema import BasePipelineConfig


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


# Register all available pipelines
def _register_pipelines():
    """Register all built-in pipelines."""
    # Import lazily to avoid circular imports and heavy dependencies
    from .krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
    from .longlive.pipeline import LongLivePipeline
    from .longlive.pipeline_vace import LongLiveVacePipeline
    from .passthrough.pipeline import PassthroughPipeline
    from .reward_forcing.pipeline import RewardForcingPipeline
    from .streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

    # Register each pipeline with its ID from its config class
    for pipeline_class in [
        LongLivePipeline,
        LongLiveVacePipeline,
        KreaRealtimeVideoPipeline,
        StreamDiffusionV2Pipeline,
        PassthroughPipeline,
        RewardForcingPipeline,
    ]:
        config_class = pipeline_class.get_config_class()
        PipelineRegistry.register(config_class.pipeline_id, pipeline_class)


# Auto-register pipelines on module import
_register_pipelines()
