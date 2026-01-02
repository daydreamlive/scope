"""Video-Depth-Anything model wrapper for depth estimation.

This module provides a wrapper around Video-Depth-Anything for consistent
depth estimation on video frames.

Reference: https://github.com/DepthAnything/Video-Depth-Anything
Paper: Video Depth Anything: Consistent Depth Estimation for Super-Long Videos (CVPR 2025)

Setup:
    The Video-Depth-Anything model code is automatically cloned from GitHub
    on first use. Model weights are downloaded from HuggingFace Hub.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Performance tuning: use streaming mode for real-time V2V
DEPTH_STREAMING_MODE = os.environ.get("DEPTH_STREAMING_MODE", "true").lower() in (
    "true",
    "1",
    "yes",
)
# Input size for depth model (lower = faster, 518 is default, try 308 or 392 for speed)
DEPTH_INPUT_SIZE = int(os.environ.get("DEPTH_INPUT_SIZE", "392"))

# Where to clone the Video-Depth-Anything repo
VDA_REPO_URL = "https://github.com/DepthAnything/Video-Depth-Anything.git"
VDA_REPO_DIR = Path(__file__).parent / "vendor" / "Video-Depth-Anything"

# Model configuration
MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "repo_id": "depth-anything/Video-Depth-Anything-Small",
        "filename": "video_depth_anything_vits.pth",
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "repo_id": "depth-anything/Video-Depth-Anything-Base",
        "filename": "video_depth_anything_vitb.pth",
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "repo_id": "depth-anything/Video-Depth-Anything-Large",
        "filename": "video_depth_anything_vitl.pth",
    },
}


def _ensure_repo_cloned():
    """Ensure Video-Depth-Anything repo is cloned."""
    if VDA_REPO_DIR.exists() and (VDA_REPO_DIR / "video_depth_anything").exists():
        return

    logger.info(f"Cloning Video-Depth-Anything repo to {VDA_REPO_DIR}...")
    VDA_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", "--depth", "1", VDA_REPO_URL, str(VDA_REPO_DIR)],
        check=True,
    )
    logger.info("Video-Depth-Anything repo cloned successfully")


def _add_repo_to_path():
    """Add the Video-Depth-Anything repo to Python path."""
    repo_path = str(VDA_REPO_DIR)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


class VideoDepthAnythingModel:
    """Video-Depth-Anything depth estimation model wrapper.

    This class provides a simple interface to load and run Video-Depth-Anything
    models for temporally consistent depth estimation on video frames.

    Args:
        encoder: Model encoder size ("vits", "vitb", or "vitl")
        device: Torch device to run inference on
        dtype: Data type for inference (default: torch.float16)
        input_size: Input size for the model (default: 518, or DEPTH_INPUT_SIZE env var)
        max_res: Maximum resolution for input (default: 1280)
        streaming: Use streaming mode for real-time processing (default: DEPTH_STREAMING_MODE)

    Example:
        >>> depth_model = VideoDepthAnythingModel(encoder="vitl", device="cuda")
        >>> depth_frames = depth_model.infer(video_frames)  # [F, H, W] in [0, 1]
    """

    def __init__(
        self,
        encoder: str = "vits",
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        input_size: int | None = None,
        max_res: int = 1280,
        streaming: bool | None = None,
    ):
        if encoder not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown encoder: {encoder}. Choose from {list(MODEL_CONFIGS.keys())}"
            )

        self.encoder = encoder
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.input_size = input_size if input_size is not None else DEPTH_INPUT_SIZE
        self.max_res = max_res
        self.streaming = streaming if streaming is not None else DEPTH_STREAMING_MODE

        self.model = None
        self.streaming_model = None  # Separate model instance for streaming
        self._model_loaded = False

        logger.info(
            f"VideoDepthAnythingModel config: encoder={encoder}, input_size={self.input_size}, "
            f"streaming={self.streaming}"
        )

    def _download_weights(self) -> Path:
        """Download model weights from HuggingFace Hub."""
        config = MODEL_CONFIGS[self.encoder]

        logger.info(f"Downloading Video-Depth-Anything weights ({self.encoder})...")
        weights_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
        )
        logger.info(f"Downloaded weights to: {weights_path}")
        return Path(weights_path)

    def load_model(self):
        """Load the Video-Depth-Anything model."""
        if self._model_loaded:
            return

        # Ensure repo is cloned and in path
        _ensure_repo_cloned()
        _add_repo_to_path()

        config = MODEL_CONFIGS[self.encoder]
        weights_path = self._download_weights()

        if self.streaming:
            # Use streaming model for real-time processing (frame-by-frame with caching)
            from video_depth_anything.video_depth_stream import (
                VideoDepthAnything as VDAStreamModel,
            )

            logger.info(
                f"Loading Video-Depth-Anything STREAMING model ({self.encoder})..."
            )

            self.streaming_model = VDAStreamModel(
                encoder=config["encoder"],
                features=config["features"],
                out_channels=config["out_channels"],
            )

            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.streaming_model.load_state_dict(state_dict)
            self.streaming_model = self.streaming_model.to(
                device=self.device, dtype=torch.float32
            )
            self.streaming_model.eval()

            logger.info("Video-Depth-Anything STREAMING model loaded successfully")
        else:
            # Use batch model for offline processing
            from video_depth_anything.video_depth import (
                VideoDepthAnything as VDAModel,
            )

            logger.info(f"Loading Video-Depth-Anything model ({self.encoder})...")

            self.model = VDAModel(
                encoder=config["encoder"],
                features=config["features"],
                out_channels=config["out_channels"],
            )

            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(state_dict)

            # Note: We use float32 for the model because the VDA code has mixed precision
            # operations that require float32 compatibility. The fp16/fp32 selection
            # is handled internally by autocast during inference.
            self.model = self.model.to(device=self.device, dtype=torch.float32)
            self.model.eval()

            logger.info("Video-Depth-Anything model loaded successfully")

        self._model_loaded = True

    @torch.no_grad()
    def infer(
        self,
        frames: torch.Tensor | np.ndarray,
        num_frames_per_batch: int = 32,
    ) -> torch.Tensor:
        """Run depth estimation on video frames.

        Args:
            frames: Video frames [F, H, W, C] in [0, 255] or [0, 1]
            num_frames_per_batch: Number of frames to process at once (default: 32)

        Returns:
            Depth maps [F, H, W] in [0, 1] range (higher = closer)
        """
        if not self._model_loaded:
            self.load_model()

        # Convert numpy to torch if needed
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        # Ensure float dtype and correct range [0, 255]
        if frames.dtype == torch.uint8:
            frames = frames.float()
        elif frames.max() <= 1.0:
            frames = frames.float() * 255.0
        else:
            frames = frames.float()

        # Handle channel dimension - expected [F, H, W, C]
        if frames.dim() == 4:
            if frames.shape[1] in [1, 3]:
                # [F, C, H, W] -> [F, H, W, C]
                frames = frames.permute(0, 2, 3, 1)

        num_frames, H, W, C = frames.shape

        # Convert to numpy array for the VDA API
        # The VDA model expects numpy array of shape [F, H, W, C]
        frames_np = frames.cpu().numpy().astype(np.uint8)

        if self.streaming and self.streaming_model is not None:
            # Use streaming mode - process frame by frame with caching
            depth_list = []
            for i in range(num_frames):
                frame_depth = self.streaming_model.infer_video_depth_one(
                    frames_np[i],
                    input_size=self.input_size,
                    device=str(self.device),
                    fp32=(self.dtype == torch.float32),
                )
                depth_list.append(frame_depth)

            depths = np.stack(depth_list, axis=0)
        else:
            # Use batch mode
            depths, _ = self.model.infer_video_depth(
                frames_np,
                target_fps=-1,  # Not used for output
                input_size=self.input_size,
                device=str(self.device),
                fp32=(self.dtype == torch.float32),
            )

        # depths is numpy array [F, H, W] of depth values
        depth = torch.from_numpy(depths).float()
        depth = depth.to(device=self.device)

        # Resize to original size if different
        if depth.shape[1] != H or depth.shape[2] != W:
            depth = depth.unsqueeze(1)  # [F, 1, H, W]
            depth = TF.interpolate(
                depth, size=(H, W), mode="bilinear", align_corners=False
            )
            depth = depth.squeeze(1)  # [F, H, W]

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def infer_from_list(
        self,
        frames: list[torch.Tensor | np.ndarray],
        num_frames_per_batch: int = 32,
    ) -> torch.Tensor:
        """Run depth estimation on a list of video frames.

        Args:
            frames: List of video frames, each [H, W, C] in [0, 255]
            num_frames_per_batch: Number of frames to process at once

        Returns:
            Depth maps [F, H, W] in [0, 1] range
        """
        # Stack frames into single tensor
        if isinstance(frames[0], np.ndarray):
            stacked = np.stack(frames, axis=0)
        else:
            stacked = torch.stack(frames, dim=0)

        return self.infer(stacked, num_frames_per_batch)

    def offload(self):
        """Offload model from GPU to free memory.

        Call this after depth estimation is complete if you need to
        free up GPU memory for other models.
        """
        if self.model is not None:
            self.model = self.model.cpu()
        if self.streaming_model is not None:
            self.streaming_model = self.streaming_model.cpu()
        torch.cuda.empty_cache()
        logger.info("Video-Depth-Anything model offloaded to CPU")

    def reset_streaming_state(self):
        """Reset the streaming model's internal cache state.

        Call this when starting a new video stream to clear cached hidden states.
        """
        if self.streaming_model is not None:
            self.streaming_model.transform = None
            self.streaming_model.frame_id_list = []
            self.streaming_model.frame_cache_list = []
            self.streaming_model.id = -1
            logger.debug("Streaming model state reset")
