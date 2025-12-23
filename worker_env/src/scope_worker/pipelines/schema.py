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
    docs_url: ClassVar[str | None] = None
    estimated_vram_gb: ClassVar[float | None] = None
    requires_models: ClassVar[bool] = False
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False

    # Mode support - override in subclasses
    supported_modes: ClassVar[list[InputMode]] = ["text"]
    default_mode: ClassVar[InputMode] = "text"

    # Prompt and temporal interpolation support
    supports_prompts: ClassVar[bool] = True
    default_temporal_interpolation_method: ClassVar[Literal["linear", "slerp"]] = (
        "slerp"
    )
    default_temporal_interpolation_steps: ClassVar[int] = 0

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
            Dict containing pipeline metadata
        """
        metadata = cls.get_pipeline_metadata()
        metadata["supported_modes"] = cls.supported_modes
        metadata["default_mode"] = cls.default_mode
        metadata["supports_prompts"] = cls.supports_prompts
        metadata["default_temporal_interpolation_method"] = (
            cls.default_temporal_interpolation_method
        )
        metadata["default_temporal_interpolation_steps"] = (
            cls.default_temporal_interpolation_steps
        )
        metadata["docs_url"] = cls.docs_url
        metadata["estimated_vram_gb"] = cls.estimated_vram_gb
        metadata["requires_models"] = cls.requires_models
        metadata["supports_lora"] = cls.supports_lora
        metadata["supports_vace"] = cls.supports_vace
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

