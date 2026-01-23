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
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Scribble pipeline.

        Args:
            config: Pipeline configuration
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

        # Model config from VACE defaults
        input_nc = getattr(config, "input_nc", 3)
        output_nc = getattr(config, "output_nc", 1)
        n_residual_blocks = getattr(config, "n_residual_blocks", 3)

        checkpoint_path = get_model_file_path(
            "VACE-Annotators/scribble/anime_style/netG_A_latest.pth"
        )

        self.model = ContourInference(
            input_nc=input_nc,
            output_nc=output_nc,
            n_residual_blocks=n_residual_blocks,
            sigmoid=True,
        )

        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            strict=True,
        )

        self.model = self.model.to(device=self.device)
        if dtype == torch.float16 and self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()

        logger.info(f"Loaded Scribble model in {time.time() - start:.3f}s")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    @torch.no_grad()
    @torch.inference_mode()
    def __call__(self, **kwargs) -> torch.Tensor:
        """Extract contour/scribble from video frames.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Contour maps as tensor in THWC format with values in [0, 1] range,
            with 3 channels (contour repeated for RGB compatibility).
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for ScribblePipeline")

        # Normalize frame sizes to handle resolution changes
        video = normalize_frame_sizes(video)

        # Process frames
        contour_frames = []

        for frame in video:
            # frame is (1, H, W, C) tensor
            frame_t = frame.squeeze(0).float()  # (H, W, C)

            # Move to device
            if frame_t.device != self.device:
                frame_t = frame_t.to(self.device)

            # Normalize to [0, 1] if needed
            if frame_t.max() > 1.0:
                frame_t = frame_t / 255.0

            # Convert to BCHW format for model: (H, W, C) -> (1, C, H, W)
            frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)

            # Convert to model dtype
            if self.dtype == torch.float16 and self.device.type == "cuda":
                frame_t = frame_t.half()

            # Run inference
            contour = self.model(frame_t)  # (1, 1, H, W)

            # Convert back to THWC: (1, 1, H, W) -> (H, W, 1)
            contour = contour.squeeze(0).permute(1, 2, 0).float()

            # Repeat to 3 channels for RGB compatibility
            contour = contour.repeat(1, 1, 3)  # (H, W, 3)

            contour_frames.append(contour)

        # Stack frames: (T, H, W, 3)
        result = torch.stack(contour_frames, dim=0)

        # Ensure output is in [0, 1] range and on CPU for downstream
        return result.clamp(0, 1).cpu()
