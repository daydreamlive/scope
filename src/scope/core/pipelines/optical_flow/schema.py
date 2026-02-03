"""Configuration schema for Optical Flow pipeline."""

from typing import Literal

from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class OpticalFlowConfig(BasePipelineConfig):
    """Configuration for Optical Flow pipeline.

    This pipeline computes optical flow between consecutive video frames using
    RAFT (Recurrent All-Pairs Field Transforms). The flow is converted to RGB
    visualization for VACE/ControlNet conditioning.

    Uses torch.compile for optimized inference.
    """

    pipeline_id = "optical-flow"
    pipeline_name = "Optical Flow"
    pipeline_description = (
        "Optical flow computation using RAFT model. "
        "Produces RGB flow visualizations for video conditioning."
    )
    docs_url = "https://pytorch.org/vision/main/models/raft.html"
    artifacts = []  # RAFT from torchvision
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # User-configurable settings
    model_size: Literal["small", "large"] = "small"
