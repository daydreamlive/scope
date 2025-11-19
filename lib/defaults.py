"""Centralized default extraction for pipelines."""

from typing import Any

# Mode constants
GENERATION_MODE_VIDEO = "video"
GENERATION_MODE_TEXT = "text"


def get_mode_defaults(pipeline_class, mode: str | None = None) -> dict[str, Any]:
    """Extract mode-specific defaults from a pipeline class.

    Args:
        pipeline_class: The pipeline class (not instance)
        mode: The mode to get defaults for (text/video). If None, uses native mode.

    Returns:
        Dictionary of mode-specific defaults including:
        - denoising_steps: list of int
        - resolution: dict with height/width
        - manage_cache: bool
        - base_seed: int
        - noise_scale: float | None
        - noise_controller: bool | None
        - Additional pipeline-specific keys (e.g. kv_cache_attention_bias)
    """
    defaults = pipeline_class.get_defaults()
    native_mode = defaults.get("native_generation_mode", GENERATION_MODE_TEXT)
    modes = defaults.get("modes", {})

    target_mode = mode or native_mode
    mode_defaults = modes.get(target_mode)

    if mode_defaults is None and target_mode != native_mode:
        mode_defaults = modes.get(native_mode)

    if mode_defaults is None:
        mode_defaults = {}

    return mode_defaults


def extract_load_params(
    pipeline_class, load_params: dict | None = None
) -> tuple[int, int, int]:
    """Extract height, width, and seed from load_params with pipeline defaults as fallback.

    Args:
        pipeline_class: The pipeline class to get defaults from
        load_params: Optional dictionary with height, width, seed overrides

    Returns:
        Tuple of (height, width, seed)
    """
    native_defaults = get_mode_defaults(pipeline_class)
    default_resolution = native_defaults.get("resolution", {})
    default_height = default_resolution.get("height", 512)
    default_width = default_resolution.get("width", 512)
    default_seed = native_defaults.get("base_seed", 42)

    if load_params:
        height = load_params.get("height", default_height)
        width = load_params.get("width", default_width)
        seed = load_params.get("seed", default_seed)
    else:
        height = default_height
        width = default_width
        seed = default_seed

    return height, width, seed


def apply_mode_defaults_to_state(
    state, pipeline_class, mode: str | None = None, kwargs: dict | None = None
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
    mode_defaults = get_mode_defaults(pipeline_class, mode)

    # Apply denoising steps if not provided
    if "denoising_step_list" not in kwargs and mode_defaults.get("denoising_steps"):
        state.set("denoising_step_list", mode_defaults["denoising_steps"])

    # For text mode, noise controls should be None (not used)
    if mode == GENERATION_MODE_TEXT:
        state.set("noise_scale", None)
        state.set("noise_controller", None)
    else:
        # For video mode, apply defaults if not provided
        if "noise_scale" not in kwargs:
            state.set("noise_scale", mode_defaults.get("noise_scale"))
        if "noise_controller" not in kwargs:
            state.set("noise_controller", mode_defaults.get("noise_controller"))
