"""Video Depth Anything Pipeline for consistent video depth estimation.

Based on Video-Depth-Anything (CVPR 2025 Highlight):
https://github.com/DepthAnything/Video-Depth-Anything
"""

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.config import get_model_file_path

from ..interface import Pipeline, Requirements
from .schema import VideoDepthAnythingConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Model configuration for Small encoder (only model supported)
MODEL_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


class VideoDepthAnythingPipeline(Pipeline):
    """Video depth estimation pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VideoDepthAnythingConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Video Depth Anything pipeline.

        Args:
            config: Pipeline configuration
            device: Target device (defaults to CUDA if available)
            dtype: Data type for model weights (default: float16)
        """
        from .modules import VideoDepthAnything

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp32 = getattr(config, "fp32", False)
        # input_size defaults to 518 (optimal for model, divisible by patch size 14)
        self.input_size = getattr(config, "input_size", 518)

        # Initialize model
        start = time.time()
        logger.info("Loading Video Depth Anything Small model...")
        checkpoint_path = get_model_file_path(
            "Video-Depth-Anything-Small/video_depth_anything_vits.pth"
        )
        self.model = VideoDepthAnything(**MODEL_CONFIG, metric=False)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model = self.model.to(device=self.device)
        if not self.fp32:
            self.model = self.model.half()
        self.model.eval()
        logger.info(f"Loaded Video Depth Anything in {time.time() - start:.3f}s")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> torch.Tensor:
        """Process video frames and return depth maps.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)
                   or tensor in BCTHW format

        Returns:
            Depth maps as tensor in THWC format with values in [0, 1] range,
            where higher values indicate greater depth (further from camera).
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError(
                "Input video cannot be None for VideoDepthAnythingPipeline"
            )

        # Convert input to numpy uint8 array
        # Frames from frame_processor are always (1, H, W, C), so we squeeze the T dimension
        frames = []
        target_shape = None
        for frame in video:
            frame_np = (
                frame.cpu().numpy()
                if isinstance(frame, torch.Tensor)
                else np.array(frame)
            )
            # Squeeze T dimension: (1, H, W, C) -> (H, W, C)
            frame_np = frame_np.squeeze(0)
            # Use first frame's shape as target for consistency
            if target_shape is None:
                target_shape = frame_np.shape
            elif frame_np.shape != target_shape:
                # Resize frame to match target shape using torch interpolate
                frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0)
                frame_tensor = torch.nn.functional.interpolate(
                    frame_tensor.float(),
                    size=(target_shape[0], target_shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).numpy()
                # Preserve original dtype
                if frames[0].dtype == np.uint8:
                    frame_np = frame_np.astype(np.uint8)
            frames.append(frame_np)
        frames = np.stack(frames, axis=0)

        # Ensure uint8 format [0, 255]
        # Note: Frames should be in RGB format [H, W, 3]
        if frames.dtype != np.uint8:
            frames = (
                (frames * 255).astype(np.uint8)
                if frames.max() <= 1.0
                else frames.astype(np.uint8)
            )

        num_frames = frames.shape[0]

        # Process frames one at a time using streaming mode
        depths = []
        cached_hidden_state_list = None
        device_str = "cuda" if self.device.type == "cuda" else "cpu"

        with torch.no_grad():
            for i in range(num_frames):
                frame = frames[i]  # [H, W, 3] in RGB format

                # Process single frame in streaming mode
                depth, cached_hidden_state_list = self.model.infer_video_depth_one(
                    frame,
                    input_size=self.input_size,
                    device=device_str,
                    fp32=self.fp32,
                    cached_hidden_state_list=cached_hidden_state_list,
                )
                depths.append(depth)

        # Stack depths: [T, H, W]
        depths = np.stack(depths, axis=0)

        # Normalize depths to [0, 1] and convert to THWC format
        depths = torch.from_numpy(depths).float()
        d_min, d_max = depths.min(), depths.max()
        depths = (
            (depths - d_min) / (d_max - d_min)
            if d_max > d_min
            else torch.zeros_like(depths)
        )
        return depths.unsqueeze(-1).repeat(1, 1, 1, 3)  # THWC with 3 channels
