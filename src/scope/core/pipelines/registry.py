"""Pipeline registry for centralized pipeline management.

This module provides a registry pattern to eliminate if/elif chains when
accessing pipelines by ID. It enables dynamic pipeline discovery and
metadata retrieval.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .interface import Pipeline


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
    def get_schema(cls, pipeline_id: str) -> dict[str, Any] | None:
        """Get schema for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline schema dictionary if found, None otherwise
        """
        pipeline_class = cls.get(pipeline_id)
        if pipeline_class is None:
            return None
        return pipeline_class.get_schema()

    @classmethod
    def list_pipelines(cls) -> list[str]:
        """Get list of all registered pipeline IDs.

        Returns:
            List of pipeline IDs
        """
        return list(cls._pipelines.keys())

    @classmethod
    def get_all_schemas(cls) -> dict[str, dict[str, Any]]:
        """Get schemas for all registered pipelines.

        Returns:
            Dictionary mapping pipeline IDs to their schemas
        """
        return {
            pipeline_id: pipeline_class.get_schema()
            for pipeline_id, pipeline_class in cls._pipelines.items()
        }


# Register all available pipelines
def _register_pipelines():
    """Register all built-in pipelines."""
    # Import lazily to avoid circular imports and heavy dependencies
    from .krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
    from .longlive.pipeline import LongLivePipeline
    from .passthrough.pipeline import PassthroughPipeline
    from .streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

    # Register each pipeline with its ID from its schema
    for pipeline_class in [
        LongLivePipeline,
        KreaRealtimeVideoPipeline,
        StreamDiffusionV2Pipeline,
        PassthroughPipeline,
    ]:
        schema = pipeline_class.get_schema()
        PipelineRegistry.register(schema["id"], pipeline_class)


# Auto-register pipelines on module import
_register_pipelines()
