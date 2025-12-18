"""Pydantic-based schema models for pipeline configuration.

This module provides Pydantic models for pipeline configuration that can be used for:
- Validation of pipeline parameters via model_validate() / model_validate_json()
- JSON Schema generation via model_json_schema()
- Type-safe configuration access
- API introspection and automatic UI generation

Pipeline-specific configs inherit from BasePipelineConfig and override defaults.
Each pipeline defines its supported modes and can provide mode-specific defaults.
"""

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

# Type alias for input modes
InputMode = Literal["text", "video"]


class ModeDefaults(BaseModel):
    """Mode-specific default values.

    These override the base config defaults when operating in a specific mode.
    Only non-None values will override the base defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # Resolution can differ per mode
    height: int | None = None
    width: int | None = None

    # Core parameters
    denoising_steps: list[int] | None = None

    # Video mode parameters
    noise_scale: float | None = None
    noise_controller: bool | None = None


class BasePipelineConfig(BaseModel):
    """Base configuration for all pipelines.

    This provides common parameters shared across all pipeline modes.
    Pipeline-specific configs inherit from this and override defaults.

    Mode support is declared via class variables:
    - supported_modes: List of modes this pipeline supports ("text", "video")
    - default_mode: The mode to use by default in the UI

    Mode-specific defaults can be provided via the get_mode_defaults() class method.
    """

    model_config = ConfigDict(extra="forbid")

    # Pipeline metadata - not configuration parameters, used for identification
    pipeline_id: ClassVar[str] = "base"
    pipeline_name: ClassVar[str] = "Base Pipeline"
    pipeline_description: ClassVar[str] = "Base pipeline configuration"
    pipeline_version: ClassVar[str] = "1.0.0"

    # Mode support - override in subclasses
    supported_modes: ClassVar[list[InputMode]] = ["text"]
    default_mode: ClassVar[InputMode] = "text"

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
    def get_mode_defaults(cls) -> dict[InputMode, ModeDefaults]:
        """Return mode-specific default overrides.

        Override in subclasses to provide different defaults per mode.
        Values in ModeDefaults override the base config defaults.

        Returns:
            Dict mapping mode name to ModeDefaults with override values
        """
        return {}

    @classmethod
    def get_defaults_for_mode(cls, mode: InputMode) -> dict[str, Any]:
        """Get effective defaults for a specific mode.

        Merges base config defaults with mode-specific overrides.

        Args:
            mode: The input mode ("text" or "video")

        Returns:
            Dict of parameter names to their effective default values
        """
        # Start with base defaults from model fields
        base_instance = cls()
        defaults = base_instance.model_dump()

        # Apply mode-specific overrides
        mode_defaults = cls.get_mode_defaults().get(mode)
        if mode_defaults:
            for field_name, value in mode_defaults.model_dump().items():
                if value is not None:
                    defaults[field_name] = value

        return defaults

    @classmethod
    def get_schema_with_metadata(cls) -> dict[str, Any]:
        """Return complete schema with pipeline metadata and JSON schema.

        This is the primary method for API/UI schema generation.

        Returns:
            Dict containing:
            - Pipeline metadata (id, name, description, version)
            - supported_modes: List of supported input modes
            - default_mode: Default input mode
            - mode_defaults: Dict of mode-specific default overrides
            - config_schema: Full JSON schema for the config model
        """
        metadata = cls.get_pipeline_metadata()
        metadata["supported_modes"] = cls.supported_modes
        metadata["default_mode"] = cls.default_mode
        metadata["config_schema"] = cls.model_json_schema()

        # Include mode-specific defaults if defined
        mode_defaults = cls.get_mode_defaults()
        if mode_defaults:
            metadata["mode_defaults"] = {
                mode: defaults.model_dump(exclude_none=True)
                for mode, defaults in mode_defaults.items()
            }

        return metadata

    def is_video_mode(self) -> bool:
        """Check if this config represents video mode.

        Returns:
            True if video mode parameters are set
        """
        return self.input_size is not None


# Concrete pipeline configurations


class LongLiveConfig(BasePipelineConfig):
    """Configuration for LongLive pipeline.

    LongLive supports both text-to-video and video-to-video modes.
    Default mode is text (T2V was the original training focus).
    """

    pipeline_id: ClassVar[str] = "longlive"
    pipeline_name: ClassVar[str] = "LongLive"
    pipeline_description: ClassVar[str] = (
        "Long-form video generation with temporal consistency"
    )

    # Mode support
    supported_modes: ClassVar[list[InputMode]] = ["text", "video"]
    default_mode: ClassVar[InputMode] = "text"

    # LongLive defaults (text mode baseline)
    height: int = Field(default=320, ge=1, description="Output height in pixels")
    width: int = Field(default=576, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
    )
    # noise_scale is None by default (text mode), overridden in video mode
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Amount of noise to add during video generation (video mode only)",
    )

    # VACE (optional reference image conditioning)
    ref_images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for VACE conditioning",
    )
    vace_context_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scaling factor for VACE hint injection (0.0 to 2.0)",
    )

    @classmethod
    def get_mode_defaults(cls) -> dict[InputMode, ModeDefaults]:
        """LongLive mode-specific defaults."""
        return {
            "text": ModeDefaults(
                # Text mode: no video input, no noise controls
                noise_scale=None,
                noise_controller=None,
            ),
            "video": ModeDefaults(
                # Video mode: requires input frames, noise controls active
                height=512,
                width=512,
                noise_scale=0.7,
                noise_controller=True,
                denoising_steps=[1000, 750],
            ),
        }


class StreamDiffusionV2Config(BasePipelineConfig):
    """Configuration for StreamDiffusion V2 pipeline.

    StreamDiffusionV2 supports both text-to-video and video-to-video modes.
    Default mode is video (V2V was the original training focus).
    """

    pipeline_id: ClassVar[str] = "streamdiffusionv2"
    pipeline_name: ClassVar[str] = "StreamDiffusion V2"
    pipeline_description: ClassVar[str] = (
        "Real-time video-to-video generation with temporal consistency"
    )

    # Mode support
    supported_modes: ClassVar[list[InputMode]] = ["text", "video"]
    default_mode: ClassVar[InputMode] = "video"

    # StreamDiffusion V2 defaults (video mode baseline since it's the default)
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

    @classmethod
    def get_mode_defaults(cls) -> dict[InputMode, ModeDefaults]:
        """StreamDiffusionV2 mode-specific defaults."""
        return {
            "text": ModeDefaults(
                # Text mode: distinct resolution, no video input, no noise controls
                height=512,
                width=512,
                noise_scale=None,
                noise_controller=None,
                denoising_steps=[1000, 750],
            ),
            "video": ModeDefaults(
                # Video mode: requires input frames, noise controls active
                noise_scale=0.7,
                noise_controller=True,
            ),
        }


class KreaRealtimeVideoConfig(BasePipelineConfig):
    """Configuration for Krea Realtime Video pipeline.

    Krea supports both text-to-video and video-to-video modes.
    Default mode is text (T2V was the original training focus).
    """

    pipeline_id: ClassVar[str] = "krea-realtime-video"
    pipeline_name: ClassVar[str] = "Krea Realtime Video"
    pipeline_description: ClassVar[str] = (
        "High-quality real-time video generation with 14B model"
    )

    # Mode support
    supported_modes: ClassVar[list[InputMode]] = ["text", "video"]
    default_mode: ClassVar[InputMode] = "text"

    # Krea defaults (text mode baseline) - distinct from LongLive (320x576)
    height: int = Field(default=320, ge=1, description="Output height in pixels")
    width: int = Field(default=576, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
    )
    # noise_scale is None by default (text mode), overridden in video mode
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Amount of noise to add during video generation (video mode only)",
    )

    @classmethod
    def get_mode_defaults(cls) -> dict[InputMode, ModeDefaults]:
        """Krea mode-specific defaults."""
        return {
            "text": ModeDefaults(
                # Text mode: no video input, no noise controls
                noise_scale=None,
                noise_controller=None,
            ),
            "video": ModeDefaults(
                # Video mode: requires input frames, noise controls active
                height=256,
                width=256,
                noise_scale=0.7,
                noise_controller=True,
                denoising_steps=[1000, 750],
            ),
        }


class RewardForcingConfig(BasePipelineConfig):
    """Configuration for RewardForcing pipeline.

    RewardForcing supports both text-to-video and video-to-video modes.
    Default mode is text (T2V was the original training focus).
    """

    pipeline_id: ClassVar[str] = "reward-forcing"
    pipeline_name: ClassVar[str] = "RewardForcing"
    pipeline_description: ClassVar[str] = (
        "Efficient streaming video generation with rewarded distribution matching distillation"
    )

    # Mode support
    supported_modes: ClassVar[list[InputMode]] = ["text", "video"]
    default_mode: ClassVar[InputMode] = "text"

    # LongLive defaults (text mode baseline)
    height: int = Field(default=320, ge=1, description="Output height in pixels")
    width: int = Field(default=576, ge=1, description="Output width in pixels")
    denoising_steps: list[int] | None = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
    )
    # noise_scale is None by default (text mode), overridden in video mode
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = Field(
        default=None,
        description="Amount of noise to add during video generation (video mode only)",
    )

    @classmethod
    def get_mode_defaults(cls) -> dict[InputMode, ModeDefaults]:
        """RewardForcing mode-specific defaults."""
        return {
            "text": ModeDefaults(
                # Text mode: no video input, no noise controls
                noise_scale=None,
                noise_controller=None,
            ),
            "video": ModeDefaults(
                # Video mode: requires input frames, noise controls active
                height=512,
                width=512,
                noise_scale=0.7,
                noise_controller=True,
                denoising_steps=[1000, 750],
            ),
        }


class PassthroughConfig(BasePipelineConfig):
    """Configuration for Passthrough pipeline (testing).

    Passthrough only supports video mode - it passes through input video frames.
    """

    pipeline_id: ClassVar[str] = "passthrough"
    pipeline_name: ClassVar[str] = "Passthrough"
    pipeline_description: ClassVar[str] = "Passthrough pipeline for testing"

    # Mode support - video only
    supported_modes: ClassVar[list[InputMode]] = ["video"]
    default_mode: ClassVar[InputMode] = "video"

    # Passthrough defaults - requires video input (distinct from StreamDiffusionV2)
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
