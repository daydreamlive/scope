"""Gray preprocessor pipeline for realtime grayscale conversion.

Based on VACE GrayAnnotator but optimized for realtime streaming.
"""

import logging
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes
from .schema import GrayConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# ITU-R BT.601 luma coefficients for RGB to grayscale
_RGB_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])


class GrayPipeline(Pipeline):
    """Grayscale conversion preprocessor optimized for realtime use."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return GrayConfig

    def __init__(
        self,
        device: torch.device | None = None,
    ):
        """Initialize the Gray pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.rgb_weights = _RGB_WEIGHTS.to(device=self.device)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Convert video frames to grayscale.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Dict with "video" key containing grayscale frames as tensor in THWC
            format with values in [0, 1] range, 3 channels for RGB compatibility.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for GrayPipeline")

        # Normalize frame sizes to handle resolution changes
        video = normalize_frame_sizes(video)

        # Stack all frames into a single tensor: (T, H, W, C)
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)

        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Batched grayscale: (T, H, W, 3) @ (3,) -> (T, H, W)
        gray = torch.matmul(frames, self.rgb_weights)

        # Expand to 3 channels for RGB compatibility: (T, H, W) -> (T, H, W, 3)
        result = gray.unsqueeze(-1).expand(-1, -1, -1, 3)

        return {"video": result.clamp(0, 1)}
