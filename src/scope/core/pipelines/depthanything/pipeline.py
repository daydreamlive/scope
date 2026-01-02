"""DepthAnything pipeline for video depth estimation.

This module provides a pipeline wrapper around Video-Depth-Anything for consistent
depth estimation on video frames. The depth maps can be used for visualization
or as conditioning signals for other pipelines.

Reference: https://github.com/DepthAnything/Video-Depth-Anything
Paper: Video Depth Anything: Consistent Depth Estimation for Super-Long Videos (CVPR 2025)

Performance notes:
    - Set streaming=True for real-time V2V (processes frame-by-frame with caching)
    - Set input_size to lower values (e.g., 308, 392) for faster inference
    - The "vits" encoder is fastest, "vitl" is most accurate
"""

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as TF

from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .model import VideoDepthAnythingModel
from .schema import DepthAnythingConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class DepthAnythingPipeline(Pipeline):
    """Pipeline for video depth estimation using Video-Depth-Anything.

    This pipeline provides temporally consistent depth estimation on video frames.
    The output can be used for visualization or as conditioning signals for
    other pipelines.

    Args:
        height: Output height in pixels
        width: Output width in pixels
        encoder: Model encoder size ("vits", "vitb", or "vitl")
        device: Torch device to run inference on
        dtype: Data type for inference (default: torch.float16)
        input_size: Input size for the model (lower = faster)
        streaming: Use streaming mode for real-time processing
        output_format: Output format ("grayscale" or "rgb")

    Example:
        >>> pipeline = DepthAnythingPipeline(encoder="vitl", device="cuda")
        >>> depth_output = pipeline(video=frames)  # returns depth visualization
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return DepthAnythingConfig

    def __init__(
        self,
        height: int = 480,
        width: int = 848,
        encoder: Literal["vits", "vitb", "vitl"] = "vits",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
        input_size: int = 392,
        streaming: bool = True,
        output_format: Literal["grayscale", "rgb"] = "grayscale",
        **kwargs,  # Accept and ignore additional parameters like 'loras', 'config', etc.
    ):
        self.height = height
        self.width = width
        self.encoder = encoder
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.input_size = input_size
        self.streaming = streaming
        self.output_format = output_format

        # Model will be loaded lazily
        self._model: VideoDepthAnythingModel | None = None
        self._model_loaded = False

        logger.info(
            f"DepthAnythingPipeline initialized: encoder={encoder}, "
            f"input_size={input_size}, streaming={streaming}, "
            f"output_format={output_format}"
        )

    def _ensure_model_loaded(self):
        """Ensure the depth model is loaded."""
        if self._model_loaded:
            return

        self._model = VideoDepthAnythingModel(
            encoder=self.encoder,
            device=self.device,
            dtype=self.dtype,
            input_size=self.input_size,
            streaming=self.streaming,
        )
        self._model.load_model()
        self._model_loaded = True

    def prepare(self, **kwargs) -> Requirements:
        """Prepare the pipeline for inference.

        Returns:
            Requirements object with input size requirements
        """
        # Load model on prepare
        self._ensure_model_loaded()
        return Requirements(input_size=4)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
        """Process video frames and return depth maps.

        Args:
            video: Input video frames as tensor [B, C, T, H, W] or list of frames
            **kwargs: Additional parameters (ignored)

        Returns:
            Depth maps in [T, H, W, C] format, values in [0, 1] range
        """
        logger.info("### DepthAnythingPipeline __call__")
        input_frames = kwargs.get("video")

        if input_frames is None:
            raise ValueError("Input 'video' cannot be None for DepthAnythingPipeline")

        self._ensure_model_loaded()

        # Handle list input - convert to tensor [F, H, W, C]
        frames = None
        if isinstance(input_frames, list):
            if len(input_frames) == 0:
                raise ValueError("Input frames list cannot be empty")

            # Check if frames are 3D [H, W, C] or 4D [T, H, W, C]
            first_frame = input_frames[0]
            if isinstance(first_frame, torch.Tensor):
                first_frame = first_frame.to(device=self.device, dtype=self.dtype)
            else:
                first_frame = torch.from_numpy(first_frame).to(device=self.device, dtype=self.dtype)

            if first_frame.dim() == 3:
                # List of [H, W, C] frames - stack them directly
                frames = torch.stack([f.to(device=self.device, dtype=self.dtype) if isinstance(f, torch.Tensor) else torch.from_numpy(f).to(device=self.device, dtype=self.dtype) for f in input_frames], dim=0)  # [F, H, W, C]
                input_height, input_width = frames.shape[1], frames.shape[2]
            elif first_frame.dim() == 4:
                # List of [T, H, W, C] frames - use preprocess_chunk
                input_frames = preprocess_chunk(
                    input_frames, self.device, self.dtype
                )
                # Continue to tensor handling below
            else:
                raise ValueError(f"Unexpected frame shape in list: {first_frame.shape}")

        # Handle tensor input format - expect [B, C, T, H, W] or [T, H, W, C] or [T, C, H, W]
        if frames is None:
            if not isinstance(input_frames, torch.Tensor):
                input_frames = torch.from_numpy(input_frames).to(device=self.device, dtype=self.dtype)
            else:
                input_frames = input_frames.to(device=self.device, dtype=self.dtype)

            if input_frames.dim() == 5:
                B, C, T, H, W = input_frames.shape
                # Store original input dimensions to preserve them in output
                input_height, input_width = H, W
                # Rearrange to [T, H, W, C] for depth model
                frames = input_frames[0].permute(1, 2, 3, 0)  # [T, H, W, C]
            elif input_frames.dim() == 4:
                # Assume [T, C, H, W] or [T, H, W, C]
                if input_frames.shape[1] == 3:
                    T, C, H, W = input_frames.shape
                    input_height, input_width = H, W
                    frames = input_frames.permute(0, 2, 3, 1)  # [T, H, W, C]
                else:
                    T, H, W, C = input_frames.shape
                    input_height, input_width = H, W
                    frames = input_frames
            else:
                raise ValueError(f"Unexpected input shape: {input_frames.shape}")

        # Convert to [0, 255] range if needed
        if frames.max() <= 1.0:
            frames = frames * 255.0

        # Run depth estimation at original input resolution
        depth = self._model.infer(frames)  # [T, H, W] in [0, 1]

        # Format output based on output_format
        # Note: VideoFrame requires RGB (3 channels), so we always output 3 channels
        if self.output_format == "grayscale":
            # [T, H, W] -> [T, H, W, 3] (replicate to 3 channels for RGB)
            output = depth.unsqueeze(-1).repeat(1, 1, 1, 3)
        elif self.output_format == "rgb":
            # [T, H, W] -> [T, H, W, 3] (replicate to 3 channels)
            output = depth.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            raise ValueError(f"Unknown output_format: {self.output_format}")

        # Output should already be at input_height x input_width (no resizing needed)
        # The depth model preserves input dimensions, so output matches input

        return postprocess_chunk(output.unsqueeze(0).permute(0, 1, 4, 2, 3))

    def reset_streaming_state(self):
        """Reset the streaming model's internal cache state.

        Call this when starting a new video stream to clear cached hidden states.
        """
        if self._model is not None:
            self._model.reset_streaming_state()

    def offload(self):
        """Offload model from GPU to free memory."""
        if self._model is not None:
            self._model.offload()
            self._model_loaded = False
