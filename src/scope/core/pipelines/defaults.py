"""Centralized default extraction for pipelines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import Pipeline, PipelineDefaults


def get_pipeline_defaults(pipeline_class: type["Pipeline"]) -> "PipelineDefaults":
    """Extract defaults from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)

    Returns:
        Typed PipelineDefaults object with pipeline configuration.
    """
    return pipeline_class.get_defaults()


def extract_load_params(
    pipeline_class: type["Pipeline"], load_params: dict | None = None
) -> tuple[int, int, int]:
    """Extract height, width, and seed from load_params with pipeline defaults as fallback.

    Args:
        pipeline_class: The pipeline class to get defaults from
        load_params: Optional dictionary with height, width, seed overrides

    Returns:
        Tuple of (height, width, seed)
    """
    defaults = get_pipeline_defaults(pipeline_class)
    default_height = defaults.resolution["height"]
    default_width = defaults.resolution["width"]
    default_seed = defaults.base_seed

    params = load_params or {}
    height = params.get("height", default_height)
    width = params.get("width", default_width)
    seed = params.get("seed", default_seed)

    return height, width, seed
