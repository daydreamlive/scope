"""
Unified artifact resolution for pipelines.

This module provides a single entry point for getting artifacts for any pipeline,
whether built-in or from a plugin. It checks the pipeline config class first,
then falls back to the legacy PIPELINE_ARTIFACTS dict.
"""

from .artifacts import Artifact


def get_artifacts_for_pipeline(pipeline_id: str) -> list[Artifact]:
    """
    Get artifacts for a pipeline, checking config class first, then legacy dict.

    Priority:
    1. Pipeline config class `artifacts` ClassVar (for plugins and migrated built-ins)
    2. Legacy PIPELINE_ARTIFACTS dict (for backwards compatibility)

    Args:
        pipeline_id: The pipeline ID to get artifacts for

    Returns:
        List of artifacts required by the pipeline, empty list if none
    """
    from scope.core.pipelines.registry import PipelineRegistry

    # Try to get from registered pipeline config class
    pipeline_class = PipelineRegistry.get(pipeline_id)
    if pipeline_class is not None:
        config_class = pipeline_class.get_config_class()
        config_artifacts = getattr(config_class, "artifacts", [])
        if config_artifacts:
            return config_artifacts

    # Fall back to legacy PIPELINE_ARTIFACTS dict
    from .pipeline_artifacts import PIPELINE_ARTIFACTS

    return PIPELINE_ARTIFACTS.get(pipeline_id, [])
