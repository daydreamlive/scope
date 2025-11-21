"""Centralized default extraction for pipelines."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers.modular_pipelines import PipelineState

    from .interface import Pipeline

logger = logging.getLogger(__name__)

# Mode constants - use these everywhere instead of magic strings
GENERATION_MODE_VIDEO = "video"
GENERATION_MODE_TEXT = "text"


def get_pipeline_schema(pipeline_class: type["Pipeline"]) -> dict[str, Any]:
    """Extract full schema from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)

    Returns:
        Complete pipeline schema dictionary with metadata and mode configurations.
    """
    return pipeline_class.get_schema()


def get_mode_config(
    pipeline_class: type["Pipeline"], mode: str | None = None
) -> dict[str, Any]:
    """Extract mode-specific configuration from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)
        mode: The mode to get config for (text/video). If None, uses native mode.

    Returns:
        Mode configuration dictionary with parameters including:
        - resolution: dict with height/width
        - denoising_steps: list of int (if applicable)
        - manage_cache: bool
        - base_seed: int
        - noise_scale: float (if applicable)
        - noise_controller: bool (if applicable)
        - Additional pipeline-specific parameters
    """
    schema = get_pipeline_schema(pipeline_class)
    native_mode = schema["native_mode"]
    mode_configs = schema["mode_configs"]

    target_mode = mode or native_mode
    mode_config = mode_configs.get(target_mode)

    if mode_config is None and target_mode != native_mode:
        logger.warning(
            f"get_mode_config: Mode '{target_mode}' not found in {pipeline_class.__name__}, "
            f"falling back to native mode '{native_mode}'"
        )
        mode_config = mode_configs.get(native_mode)

    if mode_config is None:
        raise ValueError(
            f"get_mode_config: No configuration found for mode '{target_mode}' "
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
    native_mode_config = get_mode_config(pipeline_class)
    default_height = native_mode_config["resolution"]["height"]
    default_width = native_mode_config["resolution"]["width"]
    default_seed = native_mode_config["base_seed"]

    params = load_params or {}
    height = params.get("height", default_height)
    width = params.get("width", default_width)
    seed = params.get("seed", default_seed)

    return height, width, seed


def apply_mode_defaults_to_state(
    state: "PipelineState",
    pipeline_class: type["Pipeline"],
    mode: str | None = None,
    kwargs: dict | None = None,
) -> None:
    """Apply mode-specific defaults to pipeline state.

    This consolidates the common pattern of applying defaults for denoising_steps,
    noise_scale, and noise_controller based on the current generation mode.

    Args:
        state: PipelineState object to update
        pipeline_class: The pipeline class to get defaults from
        mode: Current generation mode (text/video). If None, uses native mode.
        kwargs: Optional kwargs dict to check if parameter was explicitly provided
    """
    kwargs = kwargs or {}
    mode_config = get_mode_config(pipeline_class, mode)

    # Apply denoising steps if not provided
    denoising_steps = mode_config.get("denoising_steps")
    if "denoising_step_list" not in kwargs and denoising_steps:
        state.set("denoising_step_list", denoising_steps)

    # For text mode, noise controls should be None (not used)
    if mode == GENERATION_MODE_TEXT:
        state.set("noise_scale", None)
        state.set("noise_controller", None)
    else:
        # For video mode, apply defaults if not provided
        if "noise_scale" not in kwargs:
            state.set("noise_scale", mode_config.get("noise_scale"))
        if "noise_controller" not in kwargs:
            state.set("noise_controller", mode_config.get("noise_controller"))
