"""Pydantic configuration schema for the Moondream pipeline."""

from enum import Enum
from typing import ClassVar

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class MoondreamFeature(str, Enum):
    """Available Moondream vision features."""

    CAPTION = "caption"
    QUERY = "query"
    DETECT = "detect"
    POINT = "point"


class CaptionLength(str, Enum):
    """Caption detail level."""

    SHORT = "short"
    NORMAL = "normal"
    LONG = "long"


class MoondreamConfig(BasePipelineConfig):
    """Configuration for the Moondream vision language model pipeline."""

    pipeline_id: ClassVar[str] = "moondream"
    pipeline_name: ClassVar[str] = "Moondream"
    pipeline_description: ClassVar[str] = (
        "Vision language model: image captioning, visual Q&A, "
        "object detection, and point localization"
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    docs_url: ClassVar[str] = "https://docs.moondream.ai/transformers/"
    estimated_vram_gb: ClassVar[float] = 3.0
    supports_prompts: ClassVar[bool] = False
    modes: ClassVar[dict[str, ModeDefaults]] = {
        "video": ModeDefaults(default=True),
    }

    # --- Configuration fields (Settings panel) ---

    feature: MoondreamFeature = Field(
        default=MoondreamFeature.DETECT,
        description="Vision feature to run on each frame",
        json_schema_extra=ui_field_config(order=1, label="Feature"),
    )

    caption_length: CaptionLength = Field(
        default=CaptionLength.NORMAL,
        description="Detail level for generated captions. Only used when Feature is set to 'caption'",
        json_schema_extra=ui_field_config(order=2, label="Caption Length (caption)"),
    )

    temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for caption and query text generation",
        json_schema_extra=ui_field_config(order=3, label="Temperature (caption/query)"),
    )

    max_objects: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of objects to find. Only used when Feature is set to 'detect' or 'point'",
        json_schema_extra=ui_field_config(order=4, label="Max Objects (detect/point)"),
    )

    inference_interval: int = Field(
        default=1,
        ge=1,
        le=30,
        description="Run inference every N frames (higher = faster but less responsive)",
        json_schema_extra=ui_field_config(order=5, label="Inference Interval"),
    )

    overlay_opacity: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Opacity of annotation overlays on the video",
        json_schema_extra=ui_field_config(order=6, label="Overlay Opacity"),
    )

    font_scale: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Text size multiplier for overlay labels",
        json_schema_extra=ui_field_config(order=7, label="Font Scale"),
    )

    compile_model: bool = Field(
        default=False,
        description="Compile model on load for faster inference (slower startup)",
        json_schema_extra=ui_field_config(order=8, label="Compile Model", is_load_param=True),
    )

    # --- Input fields (Input & Controls panel) ---

    question: str = Field(
        default="What is in this image?",
        description="Question to ask about the video frame. Only used when Feature is set to 'query'",
        json_schema_extra=ui_field_config(
            order=1, label="Question (query)", category="input"
        ),
    )

    detect_object: str = Field(
        default="person",
        description="Object to find in the frame. Only used when Feature is set to 'detect' or 'point'",
        json_schema_extra=ui_field_config(
            order=2, label="Object (detect/point)", category="input"
        ),
    )
