"""
Unified artifact resolution for pipelines.

This module provides a single entry point for getting artifacts for any pipeline,
whether built-in or from a plugin.
"""

from scope.core.pipelines.artifacts import Artifact


def get_artifacts_for_pipeline(pipeline_id: str) -> list[Artifact]:
    """
    Get artifacts for a pipeline from its config class.

    Args:
        pipeline_id: The pipeline ID to get artifacts for

    Returns:
        List of artifacts required by the pipeline, empty list if none
    """
    from scope.core.pipelines.registry import PipelineRegistry

    pipeline_class = PipelineRegistry.get(pipeline_id)
    if pipeline_class is not None:
        config_class = pipeline_class.get_config_class()
        return getattr(config_class, "artifacts", [])

    return []
