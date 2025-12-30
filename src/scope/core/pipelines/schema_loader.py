"""Dynamic schema loader for YAML-based pipeline configurations.

This module provides utilities to load pipeline configuration schemas from YAML files,
automatically creating Pydantic model classes at runtime.

Pipeline developers can simply provide a schema.yaml file in their pipeline directory
instead of implementing schema.py. The loader will automatically discover and parse
these YAML files to generate the corresponding config classes.

Example schema.yaml:
    pipeline_id: "my-pipeline"
    pipeline_name: "My Pipeline"
    pipeline_description: "A great pipeline that does amazing things."
    docs_url: "https://example.com/docs"
    estimated_vram_gb: 20.0
    requires_models: true
    supports_lora: true
    supports_vace: false

    supports_cache_management: true
    supports_quantization: true
    min_dimension: 16
    modified: true

    # Instance-level field defaults
    height: 320
    width: 576
    denoising_steps: [1000, 750, 500, 250]

    # Mode configuration
    modes:
      text:
        default: true
      video:
        height: 512
        width: 512
        noise_scale: 0.7
        noise_controller: true
        denoising_steps: [1000, 750]
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .base_schema import BasePipelineConfig, ModeDefaults

logger = logging.getLogger(__name__)

# Cache for loaded config classes to avoid repeated parsing
_config_class_cache: dict[str, type[BasePipelineConfig]] = {}


# Class variables that should be set on the class, not as instance fields
CLASS_VAR_FIELDS = {
    "pipeline_id",
    "pipeline_name",
    "pipeline_description",
    "pipeline_version",
    "docs_url",
    "estimated_vram_gb",
    "requires_models",
    "supports_lora",
    "supports_vace",
    "supports_cache_management",
    "supports_kv_cache_bias",
    "supports_quantization",
    "min_dimension",
    "modified",
    "recommended_quantization_vram_threshold",
    "supports_prompts",
    "default_temporal_interpolation_method",
    "default_temporal_interpolation_steps",
    "modes",
}

# Instance fields that can be overridden with simple values
INSTANCE_FIELDS = {
    "height",
    "width",
    "denoising_steps",
    "noise_scale",
    "noise_controller",
    "input_size",
    "ref_images",
    "vace_context_scale",
    "manage_cache",
    "base_seed",
}


def _parse_modes(modes_dict: dict[str, Any]) -> dict[str, ModeDefaults]:
    """Parse modes dictionary from YAML into ModeDefaults objects.

    Args:
        modes_dict: Dictionary of mode names to their default values

    Returns:
        Dictionary of mode names to ModeDefaults instances
    """
    result = {}
    for mode_name, mode_values in modes_dict.items():
        if mode_values is None:
            mode_values = {}
        result[mode_name] = ModeDefaults(**mode_values)
    return result


def load_config_from_yaml(yaml_path: str | Path) -> type[BasePipelineConfig]:
    """Load a pipeline config class from a YAML file.

    This function parses the YAML file and dynamically creates a Pydantic
    model class that inherits from BasePipelineConfig with the specified
    class variables and field defaults.

    Args:
        yaml_path: Path to the schema.yaml file

    Returns:
        A dynamically created config class

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is malformed
        ValueError: If required fields are missing
    """
    yaml_path = Path(yaml_path)

    # Check cache first
    cache_key = str(yaml_path.resolve())
    if cache_key in _config_class_cache:
        return _config_class_cache[cache_key]

    if not yaml_path.exists():
        raise FileNotFoundError(f"Schema file not found: {yaml_path}")

    with open(yaml_path) as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

    # Validate required fields
    if "pipeline_id" not in config_data:
        raise ValueError(f"Missing required field 'pipeline_id' in {yaml_path}")

    # Generate class name from pipeline_id
    pipeline_id = config_data["pipeline_id"]
    class_name = _generate_class_name(pipeline_id)

    # Separate class variables from instance field overrides
    class_vars: dict[str, Any] = {}
    field_defaults: dict[str, Any] = {}

    for key, value in config_data.items():
        if key == "modes":
            # Special handling for modes - parse into ModeDefaults objects
            class_vars["modes"] = _parse_modes(value)
        elif key in CLASS_VAR_FIELDS:
            class_vars[key] = value
        elif key in INSTANCE_FIELDS:
            field_defaults[key] = value
        else:
            logger.warning(f"Unknown field '{key}' in {yaml_path}, ignoring")

    # Create the dynamic class with annotations for field defaults
    annotations: dict[str, Any] = {}
    for field_name, value in field_defaults.items():
        # Infer type from value
        if isinstance(value, bool):
            annotations[field_name] = bool
        elif isinstance(value, int):
            annotations[field_name] = int
        elif isinstance(value, float):
            annotations[field_name] = float
        elif isinstance(value, list):
            if value and isinstance(value[0], int):
                annotations[field_name] = list[int]
            elif value and isinstance(value[0], str):
                annotations[field_name] = list[str]
            else:
                annotations[field_name] = list
        elif value is None:
            # Keep as optional - get annotation from parent
            pass

    # Create namespace for the new class
    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        **class_vars,
        **field_defaults,
    }

    # Dynamically create the config class
    config_class = type(class_name, (BasePipelineConfig,), namespace)

    # Cache the class
    _config_class_cache[cache_key] = config_class

    return config_class


def _generate_class_name(pipeline_id: str) -> str:
    """Generate a class name from a pipeline ID.

    Converts pipeline IDs like "krea-realtime-video" to "KreaRealtimeVideoConfig".

    Args:
        pipeline_id: The pipeline identifier

    Returns:
        A PascalCase class name ending in "Config"
    """
    # Replace hyphens and underscores with spaces, title case, remove spaces
    parts = pipeline_id.replace("-", " ").replace("_", " ").split()
    pascal_case = "".join(part.capitalize() for part in parts)
    return f"{pascal_case}Config"


def discover_pipeline_schemas(pipelines_dir: str | Path) -> dict[str, type[BasePipelineConfig]]:
    """Discover all schema.yaml files in pipeline subdirectories.

    Scans the given directory for subdirectories containing schema.yaml files
    and loads each one.

    Args:
        pipelines_dir: Path to the pipelines directory

    Returns:
        Dictionary mapping pipeline IDs to their config classes
    """
    pipelines_dir = Path(pipelines_dir)
    configs = {}

    for subdir in pipelines_dir.iterdir():
        if not subdir.is_dir():
            continue

        schema_path = subdir / "schema.yaml"
        if not schema_path.exists():
            continue

        try:
            config_class = load_config_from_yaml(schema_path)
            pipeline_id = config_class.pipeline_id
            configs[pipeline_id] = config_class
            logger.debug(f"Loaded schema for pipeline '{pipeline_id}' from {schema_path}")
        except Exception as e:
            logger.warning(f"Failed to load schema from {schema_path}: {e}")

    return configs
