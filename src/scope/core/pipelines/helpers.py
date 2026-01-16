"""Helper utilities for pipeline configuration and initialization.

These helpers are used by Pipeline implementations to reduce boilerplate
in configuration and state initialization.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from diffusers.modular_pipelines import PipelineState

    from .schema import BasePipelineConfig


def initialize_state_from_config(
    state: "PipelineState",
    config: Any,
    pipeline_config: "BasePipelineConfig",
) -> None:
    """Initialize pipeline state from runtime config and pipeline defaults.

    This applies pipeline defaults from the Pydantic config model,
    with runtime config overrides when present.

    Args:
        state: PipelineState object to initialize
        config: Runtime configuration object with optional overrides
        pipeline_config: Pydantic pipeline config model with defaults
    """
    # Common state initialization
    state.set("current_start_frame", 0)

    # Apply defaults from pipeline config
    state.set("height", getattr(config, "height", pipeline_config.height))
    state.set("width", getattr(config, "width", pipeline_config.width))
    state.set(
        "manage_cache", getattr(config, "manage_cache", pipeline_config.manage_cache)
    )
    state.set("base_seed", getattr(config, "seed", pipeline_config.base_seed))

    # Optional parameters - only set if defined in pipeline config
    if pipeline_config.denoising_steps is not None:
        state.set("denoising_step_list", pipeline_config.denoising_steps)

    if pipeline_config.noise_scale is not None:
        state.set("noise_scale", pipeline_config.noise_scale)

    if pipeline_config.noise_controller is not None:
        state.set("noise_controller", pipeline_config.noise_controller)
