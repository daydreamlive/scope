"""Centralized default extraction for pipelines."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers.modular_pipelines import PipelineState

    from .interface import Pipeline
    from .schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Mode constants - use these everywhere instead of magic strings
INPUT_MODE_VIDEO = "video"
INPUT_MODE_TEXT = "text"


def get_pipeline_config(pipeline_class: type["Pipeline"]) -> "BasePipelineConfig":
    """Get the default config instance for a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)

    Returns:
        Pydantic config instance with pipeline defaults
    """
    config_class = pipeline_class.get_config_class()
    return config_class()


def resolve_input_mode(kwargs: dict[str, Any]) -> str:
    """Resolve input mode based on presence of video input.

    Mode is inferred from whether 'video' is provided in kwargs:
    - If 'video' is present and not None -> video mode
    - Otherwise -> text mode

    Args:
        kwargs: Dictionary that may contain 'video' key

    Returns:
        Resolved input mode string (text or video)

    Example:
        mode = resolve_input_mode({"video": some_tensor})
        # Returns "video"

        mode = resolve_input_mode({"prompt": "a cat"})
        # Returns "text"
    """
    if kwargs.get("video") is not None:
        return INPUT_MODE_VIDEO
    return INPUT_MODE_TEXT


def extract_load_params(
    pipeline_class: type["Pipeline"], load_params: dict | None = None
) -> tuple[int, int, int]:
    """Extract height, width, and seed from load_params with pipeline defaults as fallback.

    Uses the pipeline's default config values as fallbacks.

    Args:
        pipeline_class: The pipeline class to get defaults from
        load_params: Optional dictionary with height, width, seed overrides

    Returns:
        Tuple of (height, width, seed)
    """
    config = get_pipeline_config(pipeline_class)

    params = load_params or {}
    height = params.get("height", config.height)
    width = params.get("width", config.width)
    seed = params.get("seed", config.base_seed)

    return height, width, seed


def apply_mode_defaults_to_state(
    state: "PipelineState",
    pipeline_class: type["Pipeline"],
    mode: str | None = None,
    kwargs: dict | None = None,
) -> None:
    """Apply mode-specific defaults to pipeline state.

    This consolidates the common pattern of applying defaults for denoising_steps,
    noise_scale, and noise_controller based on the current input mode.

    Args:
        state: PipelineState object to update
        pipeline_class: The pipeline class to get defaults from
        mode: Current input mode (text/video). If None, uses text mode.
        kwargs: Optional kwargs dict to check if parameter was explicitly provided
    """
    kwargs = kwargs or {}
    config = get_pipeline_config(pipeline_class)

    # Apply denoising steps if not explicitly provided
    if "denoising_step_list" not in kwargs and config.denoising_steps:
        state.set("denoising_step_list", config.denoising_steps)

    # For text mode, noise controls should be None (not used)
    if mode == INPUT_MODE_TEXT:
        state.set("noise_scale", None)
        state.set("noise_controller", None)
    else:
        # For video mode, apply defaults if not provided
        if "noise_scale" not in kwargs and config.noise_scale is not None:
            state.set("noise_scale", config.noise_scale)
        if "noise_controller" not in kwargs and config.noise_controller is not None:
            state.set("noise_controller", config.noise_controller)
