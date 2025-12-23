"""Centralized default extraction for pipelines."""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers.modular_pipelines import PipelineState

    from .interface import Pipeline, Requirements
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


# -----------------------------------------------------------------------------
# Multi-mode pipeline helpers
# -----------------------------------------------------------------------------


def calculate_video_input_size(components_config: dict) -> int:
    """Calculate video input size from pipeline component config.

    Video input size = num_frame_per_block * vae_temporal_downsample_factor

    Args:
        components_config: Dictionary with pipeline config values (typically
            from components.config)

    Returns:
        Number of video frames required for video mode input
    """
    num_frame_per_block = components_config.get("num_frame_per_block", 3)
    vae_temporal_downsample_factor = components_config.get(
        "vae_temporal_downsample_factor", 4
    )
    return num_frame_per_block * vae_temporal_downsample_factor


def prepare_for_mode(
    pipeline_class: type["Pipeline"],
    components_config: dict,
    kwargs: dict,
    video_input_size: int | None = None,
) -> "Requirements | None":
    """Determine input requirements based on current mode.

    This is the shared implementation for multi-mode pipeline prepare() methods.
    Returns video requirements when video mode is active, None for text mode.

    Mode is determined by the presence of 'video' in kwargs (signaled by
    FrameProcessor based on the frontend's input_mode selection).

    Args:
        pipeline_class: The pipeline class (unused, kept for API compatibility)
        components_config: Dictionary with pipeline config (for calculating
            video_input_size if not provided)
        kwargs: Call kwargs - presence of 'video' key indicates video mode
        video_input_size: Override for video input size. If None, calculated
            from components_config.

    Returns:
        Requirements with input_size for video mode, None for text mode
    """
    from .interface import Requirements

    # Video mode is indicated by presence of 'video' in kwargs
    # (FrameProcessor sets this based on frontend's input_mode)
    if kwargs.get("video") is not None:
        # Calculate video input size if not provided
        if video_input_size is None:
            video_input_size = calculate_video_input_size(components_config)
        return Requirements(input_size=video_input_size)

    # No video signal means text mode
    return None


def handle_mode_transition(
    state: "PipelineState",
    vae: Any,
    first_call: bool,
    last_mode: str | None,
    kwargs: dict,
) -> tuple[bool, str]:
    """Handle mode changes and cache management for multi-mode pipelines.

    Detects mode transitions and manages cache initialization accordingly.
    On first call or mode change, sets init_cache=True and clears VAE cache.

    Args:
        state: PipelineState to update with init_cache
        vae: VAE component with clear_cache() method
        first_call: Whether this is the first call to the pipeline
        last_mode: Previous mode (None if first call)
        kwargs: Call kwargs for resolving current mode

    Returns:
        Tuple of (new_first_call, current_mode) to update pipeline state
    """
    current_mode = resolve_input_mode(kwargs)
    mode_changed = last_mode is not None and last_mode != current_mode

    if first_call or mode_changed:
        state.set("init_cache", True)
        if mode_changed:
            logger.info(
                "handle_mode_transition: Mode changed from %s to %s, resetting cache",
                last_mode,
                current_mode,
            )
            vae.clear_cache()
        first_call = False
    else:
        # This will be overridden if init_cache is passed in kwargs
        state.set("init_cache", False)

    return first_call, current_mode
