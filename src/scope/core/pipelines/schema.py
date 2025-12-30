"""Pydantic-based schema models for pipeline configuration.

This module provides Pydantic models for pipeline configuration that can be used for:
- Validation of pipeline parameters via model_validate() / model_validate_json()
- JSON Schema generation via model_json_schema()
- Type-safe configuration access
- API introspection and automatic UI generation

Pipeline-specific configs inherit from BasePipelineConfig and override defaults.
Each pipeline defines its supported modes and can provide mode-specific defaults.

Each pipeline's config class is defined in its own directory (e.g., longlive/schema.py)
and re-exported here for backwards compatibility.
"""

# Re-export base classes from base_schema for backwards compatibility
from .base_schema import BasePipelineConfig, InputMode, ModeDefaults

# Import pipeline-specific configs from their respective directories
# This provides backwards compatibility for existing imports from this module
from .krea_realtime_video.schema import KreaRealtimeVideoConfig
from .longlive.schema import LongLiveConfig
from .memflow.schema import MemFlowConfig
from .passthrough.schema import PassthroughConfig
from .reward_forcing.schema import RewardForcingConfig
from .streamdiffusionv2.schema import StreamDiffusionV2Config

# Registry of pipeline config classes
PIPELINE_CONFIGS: dict[str, type[BasePipelineConfig]] = {
    "streamdiffusionv2": StreamDiffusionV2Config,
    "longlive": LongLiveConfig,
    "krea-realtime-video": KreaRealtimeVideoConfig,
    "reward-forcing": RewardForcingConfig,
    "passthrough": PassthroughConfig,
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
