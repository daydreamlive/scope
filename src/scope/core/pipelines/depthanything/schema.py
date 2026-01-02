"""Pydantic schema for DepthAnything pipeline configuration."""

from typing import Literal

from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, height_field, width_field


class DepthAnythingConfig(BasePipelineConfig):
    """Configuration for the DepthAnything depth estimation pipeline.

    This pipeline uses Video-Depth-Anything for temporally consistent depth
    estimation on video frames. The depth maps can be used as conditioning
    signals for V2V pipelines or for visualization.

    Reference: https://github.com/DepthAnything/Video-Depth-Anything
    Paper: Video Depth Anything: Consistent Depth Estimation for Super-Long Videos (CVPR 2025)
    """

    pipeline_id = "depthanything"
    pipeline_name = "Depth Anything"
    pipeline_description = (
        "Video depth estimation pipeline using Video-Depth-Anything for "
        "temporally consistent depth maps. Useful for visualization or as "
        "conditioning signals for other pipelines."
    )
    pipeline_version = "1.0.0"

    # This pipeline doesn't need prompt input
    supports_prompts = False

    # Estimated VRAM usage (vits model - small/fast)
    estimated_vram_gb = 4.0

    # Only video mode is supported (requires input frames)
    modes = {"video": ModeDefaults(default=True, input_size=4)}

    # Resolution settings
    height: int = height_field(default=480)
    width: int = width_field(default=848)

    # Depth model settings
    encoder: Literal["vits", "vitb", "vitl"] = Field(
        default="vits",
        description="Model encoder size: vits (fastest), vitb (balanced), vitl (most accurate)",
    )

    input_size: int = Field(
        default=392,
        ge=224,
        le=518,
        description="Input size for depth model (lower = faster, 518 is default, try 308 or 392 for speed)",
    )

    streaming: bool = Field(
        default=True,
        description="Use streaming mode for real-time processing (processes frame-by-frame with caching)",
    )

    output_format: Literal["grayscale", "rgb"] = Field(
        default="grayscale",
        description="Output format: grayscale (1 channel) or rgb (3 channels)",
    )
