"""RIFE (Real-Time Intermediate Flow Estimation) frame interpolation pipeline.

Based on Practical-RIFE:
https://github.com/hzwer/Practical-RIFE
"""

import logging
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes, postprocess_chunk, preprocess_chunk
from .schema import RIFEConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class RIFEPipeline(Pipeline):
    """RIFE interpolation pipeline that doubles the frame rate of input video."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return RIFEConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the RIFE pipeline.

        Args:
            config: Pipeline configuration
            device: Target device (defaults to CUDA if available)
            dtype: Data type for processing (default: float16)
        """
        from .modules.interpolation import RIFEInterpolator

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Initialize RIFE interpolator
        logger.info("Loading RIFE HDv3 model...")
        self.rife_interpolator = RIFEInterpolator(enabled=True, device=self.device)
        logger.info("RIFE HDv3 model loaded successfully")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=12)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        input = kwargs.get("video")

        if input is None:
            raise ValueError("Input cannot be None for RIFEPipeline")

        if isinstance(input, list):
            # Normalize frame sizes to handle resolution changes
            input = normalize_frame_sizes(input)
            # Preprocess: convert list of frames to BCTHW tensor in [-1, 1] range
            input = preprocess_chunk(input, self.device, self.dtype)

        # Convert from BCTHW to THWC format for RIFE
        # First convert to BTCHW, then use postprocess_chunk to get THWC [0, 1]
        input_btchw = rearrange(input, "B C T H W -> B T C H W")
        input_thwc = postprocess_chunk(input_btchw)  # Now in THWC [0, 1] range

        # Convert to [0, 255] uint8 for RIFE interpolation
        input_uint8 = (input_thwc * 255.0).clamp(0, 255).to(torch.uint8)

        # Apply RIFE interpolation (expects THWC [0, 255] uint8, returns THWC [0, 255] uint8)
        # This doubles the frame rate: T frames -> 2*T-1 frames
        interpolated = self.rife_interpolator.interpolate(input_uint8)

        # Convert back to [0, 1] float range
        # RIFE returns THWC [0, 255] uint8, convert to THWC [0, 1] float
        interpolated_float = interpolated.float() / 255.0

        # Return THWC [0, 1] float format (same as postprocess_chunk output)
        return interpolated_float
