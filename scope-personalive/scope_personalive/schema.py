"""PersonaLive pipeline configuration schema.

This module defines the Pydantic configuration model for the PersonaLive pipeline.
"""

from typing import ClassVar, Literal

from pydantic import Field
from scope.core.pipelines.schema import BasePipelineConfig

# Type alias for input modes
InputMode = Literal["text", "video"]


class PersonaLiveConfig(BasePipelineConfig):
    """Configuration for PersonaLive portrait animation pipeline.

    PersonaLive animates a reference portrait image using driving video frames.
    It requires a reference image to be set once, then processes driving video frames.
    """

    pipeline_id: ClassVar[str] = "personalive"
    pipeline_name: ClassVar[str] = "PersonaLive"
    pipeline_description: ClassVar[str] = (
        "Real-time portrait animation from reference image and driving video"
    )
    pipeline_version: ClassVar[str] = "0.1.0"

    # Pipeline capabilities
    supports_prompts: ClassVar[bool] = False
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False
    estimated_vram_gb: ClassVar[float | None] = 8.0
    requires_models: ClassVar[bool] = True

    # Mode support - video only (requires both reference image + driving video)
    supported_modes: ClassVar[list[InputMode]] = ["video"]
    default_mode: ClassVar[InputMode] = "video"

    # PersonaLive defaults
    height: int = Field(default=512, ge=1, description="Output height in pixels")
    width: int = Field(default=512, ge=1, description="Output width in pixels")
    input_size: int | None = Field(
        default=4,
        description="Number of driving video frames per chunk",
    )

    # PersonaLive-specific parameters
    temporal_window_size: int = Field(
        default=4,
        ge=1,
        description="Temporal window size for processing",
    )
    temporal_adaptive_step: int = Field(
        default=4,
        ge=1,
        description="Temporal adaptive step for denoising",
    )
    num_inference_steps: int = Field(
        default=4,
        ge=1,
        description="Number of denoising steps (typically 4 for real-time)",
    )
