"""Centralized default extraction for pipelines."""

from typing import Any


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
    native_mode = defaults.get("native_generation_mode", "text")
    modes = defaults.get("modes", {})

    target_mode = mode or native_mode
    mode_defaults = modes.get(target_mode)

    if mode_defaults is None and target_mode != native_mode:
        mode_defaults = modes.get(native_mode)

    if mode_defaults is None:
        mode_defaults = {}

    return mode_defaults
