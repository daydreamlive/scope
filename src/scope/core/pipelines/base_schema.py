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
from scope.core.pipelines.enums import Quantization

if TYPE_CHECKING:
    from .artifacts import Artifact


# Field templates - use these to override defaults while keeping constraints/descriptions
def height_field(
    default: int = 512,
    json_schema_extra: dict[str, Any] | None = None,
) -> FieldInfo:
    """Height field with standard constraints."""
    extra = json_schema_extra or {}
    return Field(
        default=default,
        ge=1,
        description="Output video height in pixels. Higher values produce more detailed vertical resolution but reduces speed.",
        json_schema_extra=extra,
    )


def width_field(
    default: int = 512,
    json_schema_extra: dict[str, Any] | None = None,
) -> FieldInfo:
    """Width field with standard constraints."""
    extra = json_schema_extra or {}
    return Field(
        default=default,
        ge=1,
        description="Output video width in pixels. Higher values produce more detailed horizontal resolution but reduces speed.",
        json_schema_extra=extra,
    )


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


class SettingsControlType(str, Enum):
    """Special control types that require custom UI handling.

    These controls have complex UI behavior that cannot be inferred
    from the field definition alone (e.g., enabling one control affects others).
    """

    # VACE toggle with nested controls (use_input_video, context_scale)
    VACE = "vace"
    # LoRA adapters manager
    LORA = "lora"
    # Preprocessor selector
    PREPROCESSOR = "preprocessor"
    # Cache management (manage_cache toggle + reset_cache button)
    CACHE_MANAGEMENT = "cache_management"
    # Denoising step list with custom slider UI
    DENOISING_STEPS = "denoising_steps"
    # Noise controls group (noise_controller toggle + noise_scale slider)
    NOISE_CONTROLS = "noise_controls"
    # Spout sender configuration
    SPOUT_SENDER = "spout_sender"


# Type for settings panel items: either a special control type or a field name string
SettingsPanelItem = SettingsControlType | str


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
    default_temporal_interpolation_method: ClassVar[Literal["linear", "slerp"]] = (
        "slerp"
    )
    default_temporal_interpolation_steps: ClassVar[int] = 0

    # Resolution settings - use field templates for consistency
    height: int = Field(
        default=512,
        ge=1,
        description="Output video height in pixels. Higher values produce more detailed vertical resolution but reduces speed.",
        json_schema_extra={
            "ui:category": "resolution",
            "ui:order": 1,
            "ui:label": "Height",
        },
    )
    width: int = Field(
        default=512,
        ge=1,
        description="Output video width in pixels. Higher values produce more detailed horizontal resolution but reduces speed.",
        json_schema_extra={
            "ui:category": "resolution",
            "ui:order": 2,
            "ui:label": "Width",
        },
    )

    # Core parameters
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
        json_schema_extra={
            "ui:category": "cache",
            "ui:order": 1,
            "ui:label": "Manage Cache",
        },
    )
    base_seed: Annotated[int, Field(ge=0)] = Field(
        default=42,
        description="Random seed for reproducible generation. Using the same seed with the same settings will produce similar results.",
        json_schema_extra={
            "ui:category": "generation",
            "ui:order": 3,
            "ui:label": "Seed",
        },
    )
    denoising_steps: list[int] | None = Field(
        default=None,
        description="Denoising step schedule for progressive generation",
        json_schema_extra={
            "ui:category": "generation",
            "ui:order": 1,
            "ui:widget": "denoisingSteps",
            "ui:label": "Denoising Steps",
        },
    )

    # Video mode parameters (None means not applicable/text mode)
    noise_scale: Annotated[float, Field(ge=0.0, le=5.0)] | None = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Amount of noise to add during video generation (video mode only)",
        json_schema_extra={
            "ui:category": "noise",
            "ui:order": 2,
            "ui:label": "Noise Scale",
            "ui:showIf": {"field": "input_mode", "eq": "video"},
        },
    )
    noise_controller: bool | None = Field(
        default=None,
        description="Enable dynamic noise control during generation (video mode only)",
        json_schema_extra={
            "ui:category": "noise",
            "ui:order": 1,
            "ui:label": "Noise Controller",
            "ui:showIf": {"field": "input_mode", "eq": "video"},
        },
    )
    input_size: int | None = Field(
        default=1,
        ge=1,
        description="Expected input video frame count (video mode only)",
        json_schema_extra={
            "ui:hidden": True,  # Internal field, not user-facing
        },
    )

    # VACE (optional reference image conditioning)
    vace_enabled: bool = Field(
        default=False,
        description="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance.",
        json_schema_extra={
            "ui:category": "vace",
            "ui:order": 1,
            "ui:label": "VACE",
        },
    )
    vace_use_input_video: bool = Field(
        default=True,
        description="When enabled in Video input mode, the input video is used for VACE conditioning.",
        json_schema_extra={
            "ui:category": "vace",
            "ui:order": 2,
            "ui:label": "Use Input Video",
            "ui:showIf": {
                "allOf": [
                    {"field": "vace_enabled", "eq": True},
                    {"field": "input_mode", "eq": "video"},
                ]
            },
        },
    )
    ref_images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for VACE conditioning",
        json_schema_extra={
            "ui:hidden": True,  # Internal field, not user-facing
        },
    )
    vace_context_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scaling factor for VACE hint injection (0.0 to 2.0)",
        json_schema_extra={
            "ui:category": "vace",
            "ui:order": 3,
            "ui:label": "Context Scale",
            "ui:showIf": {"field": "vace_enabled", "eq": True},
        },
    )

    # Quantization (optional, only used if supports_quantization is True)
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method for the diffusion model. fp8_e4m3fn (Dynamic) reduces memory usage, but might affect performance and quality. None uses full precision and uses more memory, but does not affect performance and quality.",
        json_schema_extra={
            "ui:category": "advanced",
            "ui:order": 1,
            "ui:label": "Quantization",
        },
    )

    # KV cache attention bias (optional, only used if supports_kv_cache_bias is True)
    kv_cache_attention_bias: Annotated[float, Field(ge=0.01, le=1.0)] | None = Field(
        default=None,
        description="Controls how much to rely on past frames in the cache during generation. A lower value can help mitigate error accumulation and prevent repetitive motion. Uses log scale: 1.0 = full reliance on past frames, smaller values = less reliance on past frames. Typical values: 0.3-0.7 for moderate effect, 0.1-0.2 for strong effect.",
        json_schema_extra={
            "ui:category": "cache",
            "ui:order": 2,
            "ui:label": "KV Cache Attention Bias",
        },
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
    def _process_schema_for_ui_metadata(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Process JSON schema to extract UI metadata from json_schema_extra.

        Pydantic v2 includes json_schema_extra keys directly in the property schema.
        We move UI-related keys (ui:*) to a nested "x-ui" object for cleaner structure.

        Args:
            schema: The JSON schema dict from model_json_schema()

        Returns:
            Processed schema with UI metadata under "x-ui" key
        """
        if "properties" not in schema:
            return schema

        processed = schema.copy()
        processed["properties"] = {}

        ui_keys = {
            "ui:category",
            "ui:order",
            "ui:label",
            "ui:showIf",
            "ui:hideInModes",
            "ui:widget",
            "ui:hidden",
            "readOnly",
        }

        for prop_name, prop_schema in schema["properties"].items():
            if not isinstance(prop_schema, dict):
                processed["properties"][prop_name] = prop_schema
                continue

            prop_copy = prop_schema.copy()
            ui_metadata: dict[str, Any] = {}

            # Extract UI metadata keys
            for key in list(prop_copy.keys()):
                if key in ui_keys:
                    ui_metadata[key] = prop_copy.pop(key)

            # If we found any UI metadata, nest it under "x-ui"
            if ui_metadata:
                prop_copy["x-ui"] = ui_metadata

            processed["properties"][prop_name] = prop_copy

        return processed

    @classmethod
    def get_category_config(cls) -> dict[str, dict[str, Any]]:
        """Return category configuration for UI rendering.

        Defines section titles, order, and section-level visibility conditions.
        This is shared across all pipelines (not per-pipeline).

        Returns:
            Dict mapping category names to their configuration
        """
        return {
            "resolution": {
                "title": "Resolution",
                "icon": "ruler",
                "order": 1,
            },
            "generation": {
                "title": "Generation",
                "order": 2,
            },
            "noise": {
                "title": "Noise Control",
                "order": 3,
            },
            "vace": {
                "title": "VACE",
                "order": 4,
                "collapsible": True,
            },
            "lora": {
                "title": "LoRA",
                "order": 5,
            },
            "cache": {
                "title": "Cache",
                "order": 6,
            },
            "advanced": {
                "title": "Advanced",
                "order": 7,
                "collapsible": True,
            },
            "output": {
                "title": "Output",
                "order": 8,
            },
        }

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
        metadata["docs_url"] = cls.docs_url
        metadata["estimated_vram_gb"] = cls.estimated_vram_gb
        # Infer requires_models from artifacts if not explicitly set
        metadata["requires_models"] = cls.requires_models or bool(cls.artifacts)
        metadata["supports_lora"] = cls.supports_lora
        metadata["supports_vace"] = cls.supports_vace
        metadata["min_dimension"] = cls.min_dimension
        metadata["recommended_quantization_vram_threshold"] = (
            cls.recommended_quantization_vram_threshold
        )
        metadata["modified"] = cls.modified
        metadata["usage"] = cls.usage
        metadata["category_config"] = cls.get_category_config()

        # Generate schema and process UI metadata
        raw_schema = cls.model_json_schema()
        processed_schema = cls._process_schema_for_ui_metadata(raw_schema)
        metadata["config_schema"] = processed_schema

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
