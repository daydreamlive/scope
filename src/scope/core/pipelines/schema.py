"""Pydantic-based schema models for pipeline configuration.

This module provides Pydantic models for pipeline configuration that can be used for:
- Validation of pipeline parameters via model_validate() / model_validate_json()
- JSON Schema generation via model_json_schema()
- Type-safe configuration access
- API introspection and automatic UI generation

Pipeline-specific configs are defined via schema.yaml files in their directories.
The configs are automatically loaded and made available via this module.

To create a new pipeline:
1. Create a directory for your pipeline (e.g., my_pipeline/)
2. Add a schema.yaml file with your pipeline's configuration
3. In your pipeline.py, use:
   from ..schema_loader import get_or_create_config
   MyConfig = get_or_create_config(__file__)

Example schema.yaml:
    pipeline_id: "my-pipeline"
    pipeline_name: "My Pipeline"
    pipeline_description: "A pipeline that does X."
    height: 320
    width: 576
    modes:
      text:
        default: true
      video:
        height: 512
        width: 512
"""

from pathlib import Path

# Re-export base classes from base_schema for backwards compatibility
from .base_schema import BasePipelineConfig, InputMode, ModeDefaults
from .schema_loader import load_config_from_yaml

# Directory containing pipeline subdirectories
_PIPELINES_DIR = Path(__file__).parent

# Load pipeline configs directly from YAML files
LongLiveConfig = load_config_from_yaml(_PIPELINES_DIR / "longlive" / "schema.yaml")
PassthroughConfig = load_config_from_yaml(_PIPELINES_DIR / "passthrough" / "schema.yaml")
KreaRealtimeVideoConfig = load_config_from_yaml(_PIPELINES_DIR / "krea_realtime_video" / "schema.yaml")
RewardForcingConfig = load_config_from_yaml(_PIPELINES_DIR / "reward_forcing" / "schema.yaml")
StreamDiffusionV2Config = load_config_from_yaml(_PIPELINES_DIR / "streamdiffusionv2" / "schema.yaml")
MemFlowConfig = load_config_from_yaml(_PIPELINES_DIR / "memflow" / "schema.yaml")

# Registry of pipeline config classes
PIPELINE_CONFIGS: dict[str, type[BasePipelineConfig]] = {
    "streamdiffusionv2": StreamDiffusionV2Config,
    "longlive": LongLiveConfig,
    "krea-realtime-video": KreaRealtimeVideoConfig,
    "reward-forcing": RewardForcingConfig,
    "passthrough": PassthroughConfig,
    "memflow": MemFlowConfig,
}


def get_config_class(pipeline_id: str) -> type[BasePipelineConfig] | None:
    """Get the config class for a pipeline by ID.

    Args:
        pipeline_id: Pipeline identifier

    Returns:
        Config class if found, None otherwise
    """
    return PIPELINE_CONFIGS.get(pipeline_id)


__all__ = [
    # Base classes
    "BasePipelineConfig",
    "InputMode",
    "ModeDefaults",
    # Pipeline configs
    "StreamDiffusionV2Config",
    "LongLiveConfig",
    "KreaRealtimeVideoConfig",
    "RewardForcingConfig",
    "MemFlowConfig",
    "PassthroughConfig",
    # Registry
    "PIPELINE_CONFIGS",
    "get_config_class",
]
