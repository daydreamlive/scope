"""Pydantic-based schema models for pipeline configuration.

This module provides Pydantic models for pipeline configuration that can be used for:
- Validation of pipeline parameters via model_validate() / model_validate_json()
- JSON Schema generation via model_json_schema()
- Type-safe configuration access
- API introspection and automatic UI generation

Pipeline-specific configs inherit from BasePipelineConfig and override defaults.
"""

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class BasePipelineConfig(BaseModel):
    """Base configuration for all pipelines.

    This provides common parameters shared across all pipeline modes.
    Pipeline-specific configs inherit from this and override defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # Pipeline metadata - not configuration parameters, used for identification
    pipeline_id: ClassVar[str] = "base"
    pipeline_name: ClassVar[str] = "Base Pipeline"
    pipeline_description: ClassVar[str] = "Base pipeline configuration"
    pipeline_version: ClassVar[str] = "1.0.0"

    # Resolution settings
    height: int = Field(default=512, ge=1, description="Output height in pixels")
    width: int = Field(default=512, ge=1, description="Output width in pixels")

    # Core parameters
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
    )
    base_seed: Annotated[int, Field(ge=0)] = Field(
        default=42,
        description="Base random seed for reproducible generation",
    )
    denoising_steps: list[int] | None = Field(
        default=None,
        description="Denoising step schedule for progressive generation",
    )

    # Video mode parameters (None means not applicable/text mode)
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Amount of noise to add during video generation (video mode only)",
    )
    noise_controller: bool | None = Field(
        default=None,
        description="Enable dynamic noise control during generation (video mode only)",
    )
    input_size: int | None = Field(
        default=None,
        description="Expected input video frame count (video mode only)",
    )

    @classmethod
    def get_pipeline_metadata(cls) -> dict[str, str]:
        """Return pipeline identification metadata.

        Returns:
            Dict with id, name, description, version
        """
        return {
            "id": cls.pipeline_id,
            "name": cls.pipeline_name,
            "description": cls.pipeline_description,
            "version": cls.pipeline_version,
        }

    @classmethod
    def get_schema_with_metadata(cls) -> dict[str, Any]:
        """Return complete schema with pipeline metadata and JSON schema.

        This is the primary method for API/UI schema generation.

        Returns:
            Dict containing:
            - Pipeline metadata (id, name, description, version)
            - config_schema: Full JSON schema for the config model
        """
        metadata = cls.get_pipeline_metadata()
        metadata["config_schema"] = cls.model_json_schema()
        return metadata

    def is_video_mode(self) -> bool:
        """Check if this config represents video mode.

        Returns:
            True if video mode parameters are set
        """
        return self.input_size is not None


# Concrete pipeline configurations


class LongLiveConfig(BasePipelineConfig):
    """Configuration for LongLive pipeline."""

    pipeline_id: ClassVar[str] = "longlive"
    pipeline_name: ClassVar[str] = "LongLive"
    pipeline_description: ClassVar[str] = (
        "Long-form video generation with temporal consistency"
    )

    # LongLive defaults
    height: int = Field(default=480, ge=1, description="Output height in pixels")
    width: int = Field(default=832, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
    )


class StreamDiffusionV2Config(BasePipelineConfig):
    """Configuration for StreamDiffusion V2 pipeline."""

    pipeline_id: ClassVar[str] = "streamdiffusionv2"
    pipeline_name: ClassVar[str] = "StreamDiffusion V2"
    pipeline_description: ClassVar[str] = (
        "Real-time video-to-video generation with temporal consistency"
    )

    # StreamDiffusion V2 defaults - primarily video mode
    height: int = Field(default=512, ge=1, description="Output height in pixels")
    width: int = Field(default=512, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[750, 250],
        description="Denoising step schedule for progressive generation",
    )
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=0.7,
        description="Amount of noise to add during video generation",
    )
    noise_controller: bool | None = Field(
        default=True,
        description="Enable dynamic noise control during generation",
    )
    input_size: int | None = Field(
        default=4,
        description="Expected input video frame count",
    )


class KreaRealtimeVideoConfig(BasePipelineConfig):
    """Configuration for Krea Realtime Video pipeline."""

    pipeline_id: ClassVar[str] = "krea-realtime-video"
    pipeline_name: ClassVar[str] = "Krea Realtime Video"
    pipeline_description: ClassVar[str] = (
        "High-quality real-time video generation with 14B model"
    )

    # Krea defaults
    height: int = Field(default=480, ge=1, description="Output height in pixels")
    width: int = Field(default=832, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
    )


class PassthroughConfig(BasePipelineConfig):
    """Configuration for Passthrough pipeline (testing)."""

    pipeline_id: ClassVar[str] = "passthrough"
    pipeline_name: ClassVar[str] = "Passthrough"
    pipeline_description: ClassVar[str] = "Passthrough pipeline for testing"

    # Passthrough defaults - requires video input
    height: int = Field(default=512, ge=1, description="Output height in pixels")
    width: int = Field(default=512, ge=1, description="Output width in pixels")
    input_size: int | None = Field(
        default=4,
        description="Expected input video frame count",
    )


# Registry of pipeline config classes
PIPELINE_CONFIGS: dict[str, type[BasePipelineConfig]] = {
    "longlive": LongLiveConfig,
    "streamdiffusionv2": StreamDiffusionV2Config,
    "krea-realtime-video": KreaRealtimeVideoConfig,
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
