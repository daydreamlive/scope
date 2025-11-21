"""Centralized default extraction for pipelines."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import Pipeline, PipelineDefaults, PipelineModeConfig

logger = logging.getLogger(__name__)

# Mode constants - use these everywhere instead of magic strings
GENERATION_MODE_VIDEO = "video"
GENERATION_MODE_TEXT = "text"


def get_pipeline_defaults(pipeline_class: type["Pipeline"]) -> "PipelineDefaults":
    """Extract full mode-aware defaults from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)

    Returns:
        Typed PipelineDefaults object with native mode and mode-specific configurations.
    """
    return pipeline_class.get_defaults()


def get_mode_defaults(
    pipeline_class: type["Pipeline"], mode: str | None = None
) -> "PipelineModeConfig":
    """Extract mode-specific configuration from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)
        mode: The mode to get defaults for (text/video). If None, uses native mode.

    Returns:
        PipelineModeConfig with mode-specific defaults including:
        - denoising_steps: list of int
        - resolution: dict with height/width
        - manage_cache: bool
        - base_seed: int
        - noise_scale: float | None
        - noise_controller: bool | None
        - Additional pipeline-specific keys (e.g. kv_cache_attention_bias)
    """
    defaults = get_pipeline_defaults(pipeline_class)
    native_mode = defaults.native_generation_mode
    modes = defaults.modes

    target_mode = mode or native_mode
    mode_config = modes.get(target_mode)

    if mode_config is None and target_mode != native_mode:
        logger.warning(
            f"get_mode_defaults: Mode '{target_mode}' not found in {pipeline_class.__name__}, "
            f"falling back to native mode '{native_mode}'"
        )
        mode_config = modes.get(native_mode)

    if mode_config is None:
        raise ValueError(
            f"get_mode_defaults: No configuration found for mode '{target_mode}' "
            f"in pipeline {pipeline_class.__name__}"
        )

    return mode_config


def extract_load_params(
    pipeline_class: type["Pipeline"], load_params: dict | None = None
) -> tuple[int, int, int]:
    """Extract height, width, and seed from load_params with pipeline defaults as fallback.

    Uses the native mode's defaults as the fallback values.

    Args:
        pipeline_class: The pipeline class to get defaults from
        load_params: Optional dictionary with height, width, seed overrides

    Returns:
        Tuple of (height, width, seed)
    """
    native_mode_config = get_mode_defaults(pipeline_class)
    default_height = native_mode_config.resolution["height"]
    default_width = native_mode_config.resolution["width"]
    default_seed = native_mode_config.base_seed

    params = load_params or {}
    height = params.get("height", default_height)
    width = params.get("width", default_width)
    seed = params.get("seed", default_seed)

    return height, width, seed
