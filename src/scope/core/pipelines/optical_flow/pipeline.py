"""Optical Flow Pipeline for VACE conditioning.

Computes optical flow between consecutive frames using RAFT and converts it to
RGB visualization. Uses torch.compile for optimized inference.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torchvision.utils import flow_to_image

from ..interface import Pipeline, Requirements
from .schema import OpticalFlowConfig

# Disable CPU codegen to avoid needing MSVC on Windows
inductor_config.disable_cpp_codegen = True

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class OpticalFlowPipeline(Pipeline):
    """Optical flow pipeline for VACE conditioning.

    This pipeline computes optical flow between consecutive frames using
    RAFT (Recurrent All-Pairs Field Transforms) and converts it to RGB
    visualization for VACE/ControlNet conditioning.

    Uses torch.compile for optimized inference (~1.5x speedup over eager mode).
    The model is lazily initialized on first use.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return OpticalFlowConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
    ):
        """Initialize the optical flow pipeline.

        Args:
            config: Pipeline configuration with model_size setting
            device: Target device (defaults to CUDA if available)
        """
        from .engine import DEFAULT_HEIGHT, DEFAULT_WIDTH

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._height = DEFAULT_HEIGHT
        self._width = DEFAULT_WIDTH

        # Read settings from config
        model_size = getattr(config, "model_size", "large")

        # Model configuration
        self._use_large_model = model_size == "large"
        self._model_name = "raft_large" if self._use_large_model else "raft_small"

        self._pytorch_model = None

        start = time.time()
        model_size_str = "Large" if self._use_large_model else "Small"
        logger.info(
            f"Optical Flow pipeline initialized with RAFT {model_size_str} model "
            "(loads on first use)"
        )
        logger.info(f"Initialization time: {time.time() - start:.3f}s")

    def _ensure_pytorch_model(self):
        """Lazily initialize the PyTorch RAFT model with torch.compile.

        Returns:
            Compiled RAFT model for optimized inference
        """
        if self._pytorch_model is not None:
            return self._pytorch_model

        from .engine import load_raft_model

        model_size = "Large" if self._use_large_model else "Small"
        logger.info(f"Loading PyTorch RAFT {model_size} model...")
        start = time.time()

        model, _ = load_raft_model(self._use_large_model, device=str(self.device))

        # Compile the model for optimized inference
        logger.info(f"Compiling RAFT {model_size} model with torch.compile...")
        self._pytorch_model = torch.compile(model, backend="inductor", fullgraph=False)

        logger.info(
            f"Loaded and compiled RAFT {model_size} model in {time.time() - start:.3f}s"
        )
        return self._pytorch_model

    def _preprocess_frame_batch(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Preprocess all frames for RAFT inference in one batch operation.

        Converts frames to [0, 255] range and pads to multiple of 8 (RAFT requirement).

        Args:
            frames: All frames tensor (T, C, H, W) in [0,1] range

        Returns:
            Tuple of (preprocessed frames [T, C, H_padded, W_padded], original_h, original_w)
        """
        t, c, h, w = frames.shape

        # Convert to [0, 255] range
        frames_scaled = frames * 255.0

        # Pad to multiple of 8 if needed
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            frames_scaled = F.pad(frames_scaled, [0, pad_w, 0, pad_h])

        return frames_scaled, h, w

    def _compute_optical_flow(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Compute optical flow between two preprocessed frames.

        Args:
            frame1: First frame tensor (CHW format, [0,255], padded)
            frame2: Second frame tensor (CHW format, [0,255], padded)
            orig_h: Original height before padding
            orig_w: Original width before padding

        Returns:
            Optical flow tensor (2HW format)
        """
        model = self._ensure_pytorch_model()

        # Run PyTorch inference
        flow_predictions = model(frame1.unsqueeze(0), frame2.unsqueeze(0))
        flow = flow_predictions[-1][0]  # Last prediction, remove batch dim

        # Remove padding
        return flow[:, :orig_h, :orig_w]

    def prepare(self, **kwargs) -> Requirements:
        """Return pipeline requirements.

        Returns:
            Requirements specifying input_size needed for temporal consistency
        """
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> dict:
        """Process video frames and return optical flow visualizations.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Dict with "video" key containing flow maps as tensor in THWC format
            with values in [0, 1] range, rendered as RGB visualizations.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for OpticalFlowPipeline")

        num_frames = len(video)
        if num_frames == 0:
            raise ValueError("Input video must have at least one frame")

        # === BATCH PREPROCESSING ===
        # Convert all frames to tensor and stack at once to minimize overhead
        frame_tensors = []
        for frame in video:
            if isinstance(frame, torch.Tensor):
                ft = frame.squeeze(0)  # (1, H, W, C) -> (H, W, C)
            else:
                ft = torch.from_numpy(frame).squeeze(0)
            frame_tensors.append(ft)

        # Stack all frames: [T, H, W, C]
        frames_thwc = torch.stack(frame_tensors, dim=0)
        h, w = frames_thwc.shape[1], frames_thwc.shape[2]

        # Move to device and convert to float [0, 1] in one operation
        frames_thwc = frames_thwc.to(device=self.device, dtype=torch.float32)
        if frames_thwc.max() > 1.0:
            frames_thwc = frames_thwc / 255.0

        # Convert to CHW format for processing: [T, H, W, C] -> [T, C, H, W]
        frames_tchw = frames_thwc.permute(0, 3, 1, 2)

        # Resize all frames to flow computation size at once if needed
        if h != self._height or w != self._width:
            frames_for_flow = F.interpolate(
                frames_tchw,
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            )
        else:
            frames_for_flow = frames_tchw

        # === BATCH PREPROCESS FOR RAFT ===
        # Scale to [0, 255] and pad all frames at once
        frames_preprocessed, orig_h, orig_w = self._preprocess_frame_batch(
            frames_for_flow
        )

        # === COMPUTE OPTICAL FLOW ===
        # Preallocate output tensor [T, 3, H, W] for flow RGB
        flow_rgb_tensor = torch.empty(
            num_frames, 3, h, w, device=self.device, dtype=torch.float32
        )

        # Process consecutive frame pairs
        first_flow_idx = None
        for i in range(1, num_frames):
            prev_frame = frames_preprocessed[i - 1]
            curr_frame = frames_preprocessed[i]

            # Compute flow (frames are already preprocessed)
            flow = self._compute_optical_flow(prev_frame, curr_frame, orig_h, orig_w)

            # Convert to RGB visualization
            flow_rgb = flow_to_image(flow).float() / 255.0  # [3, flow_h, flow_w]

            # Resize to output size if needed
            if flow_rgb.shape[-2] != h or flow_rgb.shape[-1] != w:
                flow_rgb = F.interpolate(
                    flow_rgb.unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            flow_rgb_tensor[i] = flow_rgb

            # Track first computed flow
            if first_flow_idx is None:
                first_flow_idx = i

        # Fill first frame with duplicated first flow (VACE behavior)
        if first_flow_idx is not None:
            flow_rgb_tensor[0] = flow_rgb_tensor[first_flow_idx]
        else:
            # Single frame - use zero flow
            flow_rgb_tensor[0] = 0.0

        # Convert to THWC: [T, C, H, W] -> [T, H, W, C]
        flow_tensor = flow_rgb_tensor.permute(0, 2, 3, 1)

        return {"video": flow_tensor}
