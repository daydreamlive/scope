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
    """Create a mode-specific configuration dictionary.

    Args:
        resolution: Dict with 'height' and 'width' keys
        denoising_steps: List of denoising step values (None if not applicable)
        manage_cache: Whether to manage cache
        base_seed: Default random seed
        noise_scale: Noise scale value (None if not applicable)
        noise_controller: Noise controller setting (None if not applicable)
        **extra_params: Additional pipeline-specific parameters

    Returns:
        Mode configuration dictionary with all parameters
    """
    config: dict[str, Any] = {
        "resolution": resolution,
        "manage_cache": manage_cache,
        "base_seed": base_seed,
    }

    # Optional parameters - only include if not None
    if denoising_steps is not None:
        config["denoising_steps"] = denoising_steps
    if noise_scale is not None:
        config["noise_scale"] = noise_scale
    if noise_controller is not None:
        config["noise_controller"] = noise_controller

    # Add any extra parameters
    config.update(extra_params)

    return config


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
        Complete pipeline schema dictionary
    """
    return {
        "id": pipeline_id,
        "name": name,
        "description": description,
        "version": version,
        "native_mode": native_mode,
        "supported_modes": supported_modes,
        "mode_configs": mode_configs,
    }
