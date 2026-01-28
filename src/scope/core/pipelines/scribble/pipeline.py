"""Scribble preprocessor pipeline for realtime contour extraction.

Based on VACE's ScribbleAnnotator but optimized for realtime streaming.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch

from scope.core.config import get_model_file_path

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes
from .schema import ScribbleConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class ScribblePipeline(Pipeline):
    """Scribble/contour extraction preprocessor optimized for realtime use."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return ScribbleConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Scribble pipeline.

        Args:
            device: Target device (defaults to CUDA if available)
            dtype: Data type for model weights (default: float16)
        """
        from .modules import ContourInference

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Initialize model
        start = time.time()
        logger.info("Loading Scribble contour model...")

        checkpoint_path = get_model_file_path(
            "VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
        )

        self.model = ContourInference(
            input_nc=3,
            output_nc=1,
            n_residual_blocks=3,
            sigmoid=True,
        )

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            strict=True,
        )

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        logger.info(f"Loaded Scribble model in {time.time() - start:.3f}s")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, **kwargs) -> dict:
        """Extract contour/scribble from video frames.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Dict with "video" key containing contour maps as tensor in THWC format
            with values in [0, 1] range, with 3 channels for RGB compatibility.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for ScribblePipeline")

        # Normalize frame sizes to handle resolution changes
        video = normalize_frame_sizes(video)

        # Batch all frames into a single tensor for efficient inference
        # Each frame is (1, H, W, C) — squeeze, stack, then run model once
        frames = torch.stack(
            [frame.squeeze(0) for frame in video], dim=0
        )  # (T, H, W, C)

        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # Convert to BCHW: (T, H, W, C) -> (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2).to(dtype=self.dtype)

        # Run batched inference
        contours = self.model(frames)  # (T, 1, H, W)

        # Convert to THWC and expand to 3 channels for RGB compatibility
        contours = contours.float().permute(0, 2, 3, 1)  # (T, H, W, 1)
        result = contours.expand(-1, -1, -1, 3)  # (T, H, W, 3)

        return {"video": result.clamp(0, 1)}
