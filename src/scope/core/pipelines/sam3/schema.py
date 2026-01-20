"""Configuration schema for SAM3 pipeline."""

from typing import Literal

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults


class SAM3Config(BasePipelineConfig):
    """Configuration for SAM3 (Segment Anything Model 3) pipeline.

    This pipeline provides real-time object masking and tracking using Meta's
    SAM3 model. It supports text prompts for open-vocabulary segmentation,
    allowing you to segment and track objects by describing them in natural
    language (e.g., "person", "yellow school bus", "dog").

    The pipeline outputs segmentation masks that can be used for:
    - Object isolation and masking
    - Video tracking
    - Compositing and effects
    """

    pipeline_id = "sam3"
    pipeline_name = "SAM3 Segmentation"
    pipeline_description = (
        "Real-time object masking and tracking using Meta's Segment Anything Model 3. "
        "Supports text prompts for open-vocabulary segmentation and video tracking."
    )
    docs_url = "https://github.com/facebookresearch/sam3"

    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="facebook/sam3",
            files=[
                "sam3.pt",
                "config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
            ],
        ),
    ]

    supports_prompts = False
    modified = True
    estimated_vram_gb = 8.0

    modes = {"video": ModeDefaults(default=True)}

    # Segmentation prompt - what to segment/track
    segment_prompt: str = Field(
        default="person",
        description="Text description of what to segment (e.g., 'person', 'dog', 'car')",
    )

    # Output visualization mode
    output_mode: Literal["mask", "overlay", "cutout"] = Field(
        default="mask",
        description=(
            "Output format: 'mask' returns binary mask, "
            "'overlay' shows mask on original frame, "
            "'cutout' shows only the segmented object"
        ),
    )

    # Mask color for overlay mode (RGB)
    mask_color: tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="RGB color for mask overlay visualization",
    )

    # Mask opacity for overlay mode
    mask_opacity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Opacity of mask overlay (0.0 to 1.0)",
    )

    # Confidence threshold for detections
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to include a detection",
    )

    # Whether to track objects across frames
    enable_tracking: bool = Field(
        default=True,
        description="Enable object tracking across video frames",
    )
