"""OpenAPI-compatible schema utilities for pipeline metadata.

This module provides utilities for creating JSON Schema-compatible
pipeline metadata that can be used for introspection, validation,
and automatic UI generation.
"""

from typing import Any


def create_parameter_schema(
    param_type: str,
    default: Any = None,
    description: str | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    items: dict[str, Any] | None = None,
    enum: list[Any] | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """Create a JSON Schema-compatible parameter definition.

    Args:
        param_type: JSON Schema type (string, integer, number, boolean, array, object)
        default: Default value for the parameter
        description: Human-readable description
        minimum: Minimum value (for numeric types)
        maximum: Maximum value (for numeric types)
        items: Schema for array items (for array type)
        enum: Allowed values (for enum constraints)
        required: Whether this parameter is required

    Returns:
        JSON Schema parameter definition
    """
    schema: dict[str, Any] = {"type": param_type}

    if default is not None:
        schema["default"] = default
    if description is not None:
        schema["description"] = description
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if items is not None:
        schema["items"] = items
    if enum is not None:
        schema["enum"] = enum
    if required:
        schema["required"] = required

    return schema


def create_mode_config(
    resolution: dict[str, int],
    denoising_steps: list[int] | None = None,
    manage_cache: bool = True,
    base_seed: int = 42,
    noise_scale: float | None = None,
    noise_controller: bool | None = None,
    **extra_params: Any,
) -> dict[str, Any]:
    """Create a mode-specific configuration dictionary with JSON Schema format.

    Each parameter is wrapped in a JSON Schema object with type information,
    default values, and constraints following OpenAPI conventions.

    Args:
        resolution: Dict with 'height' and 'width' keys
        denoising_steps: List of denoising step values (None if not applicable)
        manage_cache: Whether to manage cache
        base_seed: Default random seed
        noise_scale: Noise scale value (None if not applicable)
        noise_controller: Noise controller setting (None if not applicable)
        **extra_params: Additional pipeline-specific parameters

    Returns:
        Mode configuration dictionary with JSON Schema-formatted parameters
    """
    config: dict[str, Any] = {
        "resolution": create_parameter_schema(
            param_type="object",
            default=resolution,
            description="Output resolution for generated frames",
        ),
        "manage_cache": create_parameter_schema(
            param_type="boolean",
            default=manage_cache,
            description="Enable automatic cache management for performance optimization",
        ),
        "base_seed": create_parameter_schema(
            param_type="integer",
            default=base_seed,
            description="Base random seed for reproducible generation",
            minimum=0,
        ),
    }

    # Optional parameters - only include if not None
    if denoising_steps is not None:
        config["denoising_steps"] = create_parameter_schema(
            param_type="array",
            default=denoising_steps,
            description="Denoising step schedule for progressive generation",
            items={"type": "integer", "minimum": 1},
        )
    if noise_scale is not None:
        config["noise_scale"] = create_parameter_schema(
            param_type="number",
            default=noise_scale,
            description="Amount of noise to add during generation",
            minimum=0.0,
            maximum=1.0,
        )
    if noise_controller is not None:
        config["noise_controller"] = create_parameter_schema(
            param_type="boolean",
            default=noise_controller,
            description="Enable dynamic noise control during generation",
        )

    # Wrap extra parameters in JSON Schema format
    for key, value in extra_params.items():
        if value is None:
            continue

        # Infer type from value
        if isinstance(value, bool):
            config[key] = create_parameter_schema(
                param_type="boolean",
                default=value,
            )
        elif isinstance(value, int):
            config[key] = create_parameter_schema(
                param_type="integer",
                default=value,
            )
        elif isinstance(value, float):
            config[key] = create_parameter_schema(
                param_type="number",
                default=value,
            )
        elif isinstance(value, str):
            config[key] = create_parameter_schema(
                param_type="string",
                default=value,
            )
        elif isinstance(value, list):
            config[key] = create_parameter_schema(
                param_type="array",
                default=value,
            )
        elif isinstance(value, dict):
            config[key] = create_parameter_schema(
                param_type="object",
                default=value,
            )
        else:
            # For unknown types, store as-is
            config[key] = value

    return config


def compute_capabilities(
    supported_modes: list[str],
    mode_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute pipeline capabilities from mode configurations.

    This function derives UI-relevant capabilities from the schema structure,
    eliminating the need for frontend logic to perform these derivations.

    Works with JSON Schema-formatted parameters (extracts from schema objects).

    Args:
        supported_modes: List of supported generation modes
        mode_configs: Dict mapping mode names to their configurations

    Returns:
        Dictionary of computed capabilities including:
        - hasGenerationModeControl: Whether pipeline supports mode switching
        - hasNoiseControls: Whether any mode supports noise controls
        - showNoiseControlsInText: Whether text mode has noise controls
        - showNoiseControlsInVideo: Whether video mode has noise controls
        - hasCacheManagement: Whether any mode supports cache management
        - requiresVideoInVideoMode: Whether video mode requires video input
    """
    text_config = mode_configs.get("text", {})
    video_config = mode_configs.get("video", {})

    # Check if noise controls are available in each mode
    # Parameters are now JSON Schema objects, so check for their presence
    show_noise_controls_in_text = (
        text_config.get("noise_scale") is not None
        or text_config.get("noise_controller") is not None
    )
    show_noise_controls_in_video = (
        video_config.get("noise_scale") is not None
        or video_config.get("noise_controller") is not None
    )

    # Derive high-level capabilities
    has_generation_mode_control = len(supported_modes) > 1
    has_noise_controls = show_noise_controls_in_text or show_noise_controls_in_video
    has_cache_management = (
        text_config.get("manage_cache") is not None
        or video_config.get("manage_cache") is not None
    )

    # Video mode requires video input if it has input_size specified
    requires_video_in_video_mode = video_config.get("input_size") is not None

    return {
        "hasGenerationModeControl": has_generation_mode_control,
        "hasNoiseControls": has_noise_controls,
        "showNoiseControlsInText": show_noise_controls_in_text,
        "showNoiseControlsInVideo": show_noise_controls_in_video,
        "hasCacheManagement": has_cache_management,
        "requiresVideoInVideoMode": requires_video_in_video_mode,
    }


def create_pipeline_schema(
    pipeline_id: str,
    name: str,
    description: str,
    native_mode: str,
    supported_modes: list[str],
    mode_configs: dict[str, dict[str, Any]],
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Create a complete pipeline metadata schema.

    This creates an OpenAPI-compatible schema that includes:
    - Pipeline identification and metadata
    - Supported generation modes
    - Mode-specific parameter configurations
    - Computed capabilities for UI generation
    - JSON Schema-compatible parameter definitions

    Args:
        pipeline_id: Unique identifier for the pipeline
        name: Human-readable pipeline name
        description: Description of pipeline capabilities
        native_mode: Native generation mode (text or video)
        supported_modes: List of supported modes
        mode_configs: Dict mapping mode names to their configurations
        version: Pipeline version string

    Returns:
        Complete pipeline schema dictionary with computed capabilities
    """
    # Compute capabilities from mode configs
    capabilities = compute_capabilities(supported_modes, mode_configs)

    return {
        "id": pipeline_id,
        "name": name,
        "description": description,
        "version": version,
        "native_mode": native_mode,
        "supported_modes": supported_modes,
        "mode_configs": mode_configs,
        "capabilities": capabilities,
    }
