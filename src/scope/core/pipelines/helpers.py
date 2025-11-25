"""Helper utilities for pipeline configuration and initialization.

These helpers are used by Pipeline implementations to reduce boilerplate
in configuration and state initialization.
"""

from typing import Any

from diffusers.modular_pipelines import PipelineState

from .schema import create_mode_config


def create_mode_configs(
    shared: dict[str, Any],
    text_overrides: dict[str, Any] | None = None,
    video_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Helper to create mode configs with shared values and mode-specific overrides.

    Args:
        shared: Common configuration values shared across both modes
        text_overrides: Text-mode specific overrides (default: empty)
        video_overrides: Video-mode specific overrides (default: empty)

    Returns:
        Dictionary with 'text' and 'video' mode configurations
    """
    text_overrides = text_overrides or {}
    video_overrides = video_overrides or {}

    from .defaults import GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO

    return {
        GENERATION_MODE_TEXT: create_mode_config(**{**shared, **text_overrides}),
        GENERATION_MODE_VIDEO: create_mode_config(**{**shared, **video_overrides}),
    }


def build_pipeline_schema(
    pipeline_id: str,
    name: str,
    description: str,
    native_mode: str,
    shared: dict[str, Any],
    text_overrides: dict[str, Any] | None = None,
    video_overrides: dict[str, Any] | None = None,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Build complete pipeline schema with native mode and mode configs.

    This helper reduces boilerplate by combining mode config creation with
    schema construction.

    Args:
        pipeline_id: Unique identifier for the pipeline
        name: Human-readable pipeline name
        description: Description of pipeline capabilities
        native_mode: Native generation mode (text or video)
        shared: Common configuration values shared across both modes
        text_overrides: Text-mode specific overrides (default: empty)
        video_overrides: Video-mode specific overrides (default: empty)
        version: Pipeline version string

    Returns:
        Complete pipeline schema dictionary

    Example:
        return build_pipeline_schema(
            pipeline_id="streamdiffusionv2",
            name="StreamDiffusion V2",
            description="Video-to-video generation with temporal consistency",
            native_mode=GENERATION_MODE_VIDEO,
            shared={"manage_cache": True, "base_seed": 42},
            text_overrides={
                "resolution": {"height": 512, "width": 512},
                "denoising_steps": [1000, 750],
            },
            video_overrides={
                "resolution": {"height": 512, "width": 512},
                "denoising_steps": [750, 250],
                "noise_scale": 0.7,
                "noise_controller": True,
            },
        )
    """
    from .defaults import GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO
    from .schema import create_pipeline_schema

    mode_configs = create_mode_configs(shared, text_overrides, video_overrides)

    supported_modes = [GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO]

    return create_pipeline_schema(
        pipeline_id=pipeline_id,
        name=name,
        description=description,
        native_mode=native_mode,
        supported_modes=supported_modes,
        mode_configs=mode_configs,
        version=version,
    )


def initialize_state_from_config(
    state: PipelineState,
    config: Any,
    mode_config: dict[str, Any],
) -> None:
    """Initialize pipeline state from config and mode-specific defaults.

    This iterates through the mode configuration and sets state values,
    using config overrides when present.

    Args:
        state: PipelineState object to initialize
        config: Configuration object with optional overrides
        mode_config: Mode-specific configuration dictionary
    """
    # Common state initialization
    state.set("current_start_frame", 0)

    # Iterate through mode config and apply to state
    for key, default_value in mode_config.items():
        if key == "resolution":
            # Special handling for resolution dict
            state.set("height", getattr(config, "height", default_value["height"]))
            state.set("width", getattr(config, "width", default_value["width"]))
        elif key == "base_seed":
            # Map base_seed from config to state, checking for 'seed' attribute
            state.set("base_seed", getattr(config, "seed", default_value))
        else:
            # Direct mapping for other parameters
            state.set(key, getattr(config, key, default_value))
