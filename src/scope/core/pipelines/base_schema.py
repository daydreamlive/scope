"""Base Pydantic schema models for pipeline configuration.

This module provides the base Pydantic models for pipeline configuration.
Pipeline-specific configs should import from this module to avoid circular imports.

Pipeline-specific configs inherit from BasePipelineConfig and override defaults.
Each pipeline defines its supported modes and can provide mode-specific defaults.

Child classes can override field defaults with type-annotated assignments:
    height: int = 320
    width: int = 576
    denoising_steps: list[int] = [1000, 750, 500, 250]

For pipelines that support controller input (WASD/mouse), include a ctrl_input field:
    ctrl_input: CtrlInput | None = None
"""

from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

# Re-export CtrlInput for convenient import by pipeline schemas
from scope.core.pipelines.controller import CtrlInput as CtrlInput  # noqa: PLC0414

if TYPE_CHECKING:
    from .artifacts import Artifact


# Field templates - use these to override defaults while keeping constraints/descriptions
def height_field(default: int = 512) -> FieldInfo:
    """Height field with standard constraints."""
    return Field(default=default, ge=1, description="Output height in pixels")


def width_field(default: int = 512) -> FieldInfo:
    """Width field with standard constraints."""
    return Field(default=default, ge=1, description="Output width in pixels")


def denoising_steps_field(default: list[int] | None = None) -> FieldInfo:
    """Denoising steps field."""
    return Field(
        default=default,
        description="Denoising step schedule for progressive generation",
    )


def noise_scale_field(default: float | None = None) -> FieldInfo:
    """Noise scale field with constraints."""
    return Field(
        default=default,
        ge=0.0,
        le=1.0,
        description="Amount of noise to add during video generation (video mode only)",
    )


def noise_controller_field(default: bool | None = None) -> FieldInfo:
    """Noise controller field."""
    return Field(
        default=default,
        description="Enable dynamic noise control during generation (video mode only)",
    )


def input_size_field(default: int | None = 1) -> FieldInfo:
    """Input size field with constraints."""
    return Field(
        default=default,
        ge=1,
        description="Expected input video frame count (video mode only)",
    )


def ref_images_field(default: list[str] | None = None) -> FieldInfo:
    """Reference images field for VACE."""
    return Field(
        default=default,
        description="List of reference image paths for VACE conditioning",
    )


def vace_context_scale_field(default: float = 1.0) -> FieldInfo:
    """VACE context scale field with constraints."""
    return Field(
        default=default,
        ge=0.0,
        le=2.0,
        description="Scaling factor for VACE hint injection (0.0 to 2.0)",
    )


# Type alias for input modes
InputMode = Literal["text", "video"]


class UsageType(str, Enum):
    """Usage types for pipelines."""

    PREPROCESSOR = "preprocessor"
    POSTPROCESSOR = "postprocessor"


class ModeDefaults(BaseModel):
    """Mode-specific default values.

    Use this to define mode-specific overrides in pipeline schemas.
    Only include fields that differ from base defaults.
    Set default=True to mark the default mode.

    Example:
        modes = {
            "text": ModeDefaults(default=True),
            "video": ModeDefaults(
                height=512,
                width=512,
                noise_scale=0.7,
                noise_controller=True,
            ),
        }
    """

    model_config = ConfigDict(extra="forbid")

    # Whether this is the default mode
    default: bool = False

    # Resolution can differ per mode
    height: int | None = None
    width: int | None = None

    # Core parameters
    denoising_steps: list[int] | None = None

    # Video mode parameters
    noise_scale: float | None = None
    noise_controller: bool | None = None
    input_size: int | None = None

    # Temporal interpolation
    default_temporal_interpolation_steps: int | None = None


class BasePipelineConfig(BaseModel):
    """Base configuration for all pipelines.

    This provides common parameters shared across all pipeline modes.
    Pipeline-specific configs inherit from this and override defaults.

    Mode support is declared via the `modes` class variable:
        modes = {
            "text": ModeDefaults(default=True),
            "video": ModeDefaults(
                height=512,
                width=512,
                noise_scale=0.7,
            ),
        }

    Only include fields that differ from base defaults.
    Use default=True to mark the default mode.
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
    artifacts: ClassVar[list["Artifact"]] = []
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False

    # UI capability metadata - tells frontend what controls to show
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = False
    min_dimension: ClassVar[int] = 1
    # Whether this pipeline contains modifications based on the original project
    modified: ClassVar[bool] = False
    # Recommended quantization based on VRAM: if user's VRAM > this threshold (GB),
    # quantization=null is recommended, otherwise fp8_e4m3fn is recommended.
    # None means no specific recommendation (pipeline doesn't benefit from quantization).
    recommended_quantization_vram_threshold: ClassVar[float | None] = None
    # Usage types: list of usage types indicating how this pipeline can be used.
    # Pipelines are always available in the pipeline select dropdown.
    # Only preprocessors need to explicitly define usage = [UsageType.PREPROCESSOR]
    # to appear in the preprocessor dropdown.
    usage: ClassVar[list[UsageType]] = []

    # Mode configuration - keys are mode names, values are ModeDefaults with field overrides
    # Use default=True to mark the default mode. Only include fields that differ from base.
    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}

    # Prompt and temporal interpolation support
    supports_prompts: ClassVar[bool] = True
    default_temporal_interpolation_method: ClassVar[
        Literal["linear", "slerp"] | None
    ] = "slerp"
    default_temporal_interpolation_steps: ClassVar[int | None] = 0
    default_spatial_interpolation_method: ClassVar[
        Literal["linear", "slerp"] | None
    ] = "linear"

    # Resolution settings - use field templates for consistency
    height: int = height_field()
    width: int = width_field()

    # Core parameters
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
    )
    base_seed: Annotated[int, Field(ge=0)] = Field(
        default=42,
        description="Base random seed for reproducible generation",
    )
    denoising_steps: list[int] | None = denoising_steps_field()

    # Video mode parameters (None means not applicable/text mode)
    noise_scale: Annotated[float, Field(ge=0.0, le=1.0)] | None = noise_scale_field()
    noise_controller: bool | None = noise_controller_field()
    input_size: int | None = input_size_field()

    # VACE (optional reference image conditioning)
    ref_images: list[str] | None = ref_images_field()
    vace_context_scale: float = vace_context_scale_field()

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
    def get_supported_modes(cls) -> list[str]:
        """Return list of supported mode names."""
        return list(cls.modes.keys())

    @classmethod
    def get_default_mode(cls) -> str:
        """Return the default mode name.

        Returns the mode marked with default=True, or the first mode if none marked.
        """
        for mode_name, mode_config in cls.modes.items():
            if mode_config.default:
                return mode_name
        # Fallback to first mode if none marked as default
        return next(iter(cls.modes.keys()))

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

        # Apply mode-specific overrides (excluding None values and the "default" flag)
        mode_config = cls.modes.get(mode)
        if mode_config:
            for field_name, value in mode_config.model_dump(
                exclude={"default"}
            ).items():
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
        metadata["supported_modes"] = cls.get_supported_modes()
        metadata["default_mode"] = cls.get_default_mode()
        metadata["supports_prompts"] = cls.supports_prompts
        metadata["default_temporal_interpolation_method"] = (
            cls.default_temporal_interpolation_method
        )
        metadata["default_temporal_interpolation_steps"] = (
            cls.default_temporal_interpolation_steps
        )
        metadata["default_spatial_interpolation_method"] = (
            cls.default_spatial_interpolation_method
        )
        metadata["docs_url"] = cls.docs_url
        metadata["estimated_vram_gb"] = cls.estimated_vram_gb
        # Infer requires_models from artifacts if not explicitly set
        metadata["requires_models"] = cls.requires_models or bool(cls.artifacts)
        metadata["supports_lora"] = cls.supports_lora
        metadata["supports_vace"] = cls.supports_vace
        metadata["supports_cache_management"] = cls.supports_cache_management
        metadata["supports_kv_cache_bias"] = cls.supports_kv_cache_bias
        metadata["supports_quantization"] = cls.supports_quantization
        metadata["min_dimension"] = cls.min_dimension
        metadata["recommended_quantization_vram_threshold"] = (
            cls.recommended_quantization_vram_threshold
        )
        metadata["modified"] = cls.modified
        # Convert UsageType enum values to strings for JSON serialization
        metadata["usage"] = [usage.value for usage in cls.usage] if cls.usage else []
        metadata["config_schema"] = cls.model_json_schema()

        # Include mode-specific defaults (excluding None values and the "default" flag)
        mode_defaults = {}
        for mode_name, mode_config in cls.modes.items():
            overrides = mode_config.model_dump(exclude={"default"}, exclude_none=True)
            if overrides:
                mode_defaults[mode_name] = overrides
        if mode_defaults:
            metadata["mode_defaults"] = mode_defaults

        return metadata

    def is_video_mode(self) -> bool:
        """Check if this config represents video mode.

        Returns:
            True if video mode parameters are set
        """
        return self.input_size is not None
