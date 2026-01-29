"""Configuration schema for Image Filter post processor pipeline.

Inspired by ComfyUI-Purz Image Filter Live node.
"""

from enum import Enum

from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class FilterPreset(str, Enum):
    """Available filter presets."""

    CUSTOM = "custom"
    # Cinematic
    CINEMATIC = "cinematic"
    BLOCKBUSTER = "blockbuster"
    NOIR = "noir"
    SCIFI = "scifi"
    HORROR = "horror"
    # Film
    KODACHROME = "kodachrome"
    POLAROID = "polaroid"
    VINTAGE_70S = "vintage_70s"
    CROSS_PROCESS = "cross_process"
    # Portrait
    SOFT_PORTRAIT = "soft_portrait"
    WARM_PORTRAIT = "warm_portrait"
    COOL_PORTRAIT = "cool_portrait"
    # Landscape
    VIVID_NATURE = "vivid_nature"
    GOLDEN_HOUR = "golden_hour"
    MISTY_MORNING = "misty_morning"
    # Black & White
    BW_HIGH_CONTRAST = "bw_high_contrast"
    BW_FILM_NOIR = "bw_film_noir"
    BW_SOFT = "bw_soft"
    # Mood
    DREAMY = "dreamy"
    MOODY_BLUE = "moody_blue"
    WARM_SUNSET = "warm_sunset"
    # Creative
    NEON_NIGHTS = "neon_nights"
    RETRO_GAMING = "retro_gaming"
    DUOTONE_POP = "duotone_pop"
    VHS_TAPE = "vhs_tape"
    GLITCH_ART = "glitch_art"
    CYBERPUNK = "cyberpunk"
    COMIC_BOOK = "comic_book"
    SKETCH = "sketch"
    THERMAL = "thermal"


class ImageFilterConfig(BasePipelineConfig):
    """Configuration for Image Filter post processor pipeline.

    A real-time image filter pipeline that applies color grading and effects
    to video frames. Supports brightness, contrast, saturation, exposure,
    vignette, grain, sharpening, and more.

    Select a preset for quick looks, or use "Custom" to adjust individual parameters.
    When a preset is selected, individual parameter sliders are ignored.

    Inspired by ComfyUI-Purz Image Filter Live node.
    """

    pipeline_id = "image_filter"
    pipeline_name = "Image Filter"
    pipeline_description = (
        "Real-time image filter pipeline for color grading and effects. "
        "Apply brightness, contrast, saturation, vignette, grain, and more."
    )
    supports_prompts = False
    modified = True

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # === Preset Selection ===
    # Note: Preset is now controlled via the Effect Layers custom component
    preset: FilterPreset = Field(
        default=FilterPreset.CUSTOM,
        description="Select a preset look, or 'Custom' to use individual sliders",
    )

    # === Effect Layers (Custom Web Component) ===
    effect_layers: str = Field(
        default='{"preset": "custom", "layers": []}',
        description="Preset and layer-based effect configuration (JSON)",
        json_schema_extra=ui_field_config(
            order=0,
            component="custom",
            script_url="/api/v1/pipelines/image_filter/static/effect-layers.js",
            element_name="effect-layers",
            label="Image Filter",
        ),
    )

    # === Basic Adjustments ===
    # Note: These fields are managed by the Effect Layers custom component.
    # They have no ui_field_config so they don't appear as individual sliders.
    brightness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust overall brightness (-1 to 1)",
    )

    contrast: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust contrast (-1 to 1)",
    )

    saturation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust color saturation (-1 = grayscale, 1 = vibrant)",
    )

    exposure: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Adjust exposure in stops (-2 to 2)",
    )

    gamma: float = Field(
        default=1.0,
        ge=0.2,
        le=3.0,
        description="Adjust gamma curve (0.2 to 3.0, 1.0 = no change)",
    )

    # === Color Manipulation ===
    hue_shift: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Shift hue around color wheel (-1 to 1, wraps around)",
    )

    temperature: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Color temperature (-1 = cool/blue, 1 = warm/orange)",
    )

    tint: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Tint adjustment (-1 = green, 1 = magenta)",
    )

    vibrance: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Selective saturation boost for muted colors",
    )

    # === Tone Adjustments ===
    highlights: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust highlight tones",
    )

    shadows: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Adjust shadow tones",
    )

    # === Detail ===
    sharpen: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sharpen amount (0 = none, 2 = strong)",
    )

    blur: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Blur radius in pixels (0 = none)",
    )

    # === Stylistic Effects ===
    vignette: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Vignette intensity (0 = none, 1 = strong)",
    )

    grain: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Film grain intensity (0 = none, 0.5 = strong)",
    )

    sepia: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sepia tone intensity (0 = none, 1 = full)",
    )

    invert: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Color inversion amount (0 = normal, 1 = fully inverted)",
    )

    posterize_levels: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Posterize color levels (0 = disabled, 2-32 = number of levels)",
    )

    # === Creative Effects ===
    chromatic_aberration: float = Field(
        default=0.0,
        ge=0.0,
        le=20.0,
        description="RGB channel separation in pixels (0 = none)",
    )

    glitch: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Glitch effect intensity with horizontal band displacement",
    )

    pixelate: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Pixelation block size (0 = disabled)",
    )

    emboss: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Emboss/relief effect intensity",
    )

    edge_detect: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Edge detection intensity (outlines)",
    )

    halftone: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Halftone dot size (0 = disabled)",
    )

    scanlines: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CRT scanline effect intensity",
    )

    rgb_shift: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Shift RGB channels vertically (-1 to 1)",
    )

    mirror: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Horizontal mirror/kaleidoscope effect",
    )

    noise: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Color noise overlay intensity",
    )
