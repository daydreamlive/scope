"""Optical Flow Pipeline for VACE conditioning.

Computes optical flow between consecutive frames using RAFT and converts it to
RGB visualization. Uses TensorRT acceleration when available and enabled,
otherwise falls back to PyTorch RAFT.
"""

import logging
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torchvision.utils import flow_to_image

from ..interface import Pipeline, Requirements
from .download import get_engine_path, get_models_dir, get_onnx_path
from .schema import OpticalFlowConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _is_tensorrt_available() -> bool:
    """Check if TensorRT and polygraphy are available."""
    try:
        import tensorrt  # noqa: F401
        from polygraphy.backend.trt import engine_from_bytes  # noqa: F401

        return True
    except ImportError:
        return False


class OpticalFlowPipeline(Pipeline):
    """Optical flow pipeline for VACE conditioning.

    This pipeline computes optical flow between consecutive frames using
    RAFT (Recurrent All-Pairs Field Transforms) and converts it to RGB
    visualization for VACE/ControlNet conditioning.

    When TensorRT is available and enabled via config, uses TensorRT
    acceleration. Otherwise falls back to PyTorch RAFT inference.

    The TensorRT engine is lazily initialized on first use, automatically
    exporting RAFT from torchvision and compiling it to a GPU-specific engine.
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
            config: Pipeline configuration with model_size and use_tensorrt settings
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

        # Read settings from config (set by pipeline_manager from load_params)
        model_size = getattr(config, "model_size", "large")
        use_tensorrt_config = getattr(config, "use_tensorrt", False)

        # Model configuration
        self._use_large_model = model_size == "large"
        self._model_name = "raft_large" if self._use_large_model else "raft_small"

        # RAFT Large requires FP32 for TensorRT (FP16 causes NaN)
        # RAFT Small can use FP16
        self._fp16 = not self._use_large_model

        # Determine backend (config setting AND runtime availability)
        self._use_tensorrt = use_tensorrt_config and _is_tensorrt_available()

        self._trt_engine = None
        self._pytorch_model = None
        self._is_cuda_available = torch.cuda.is_available()

        start = time.time()
        backend = "TensorRT" if self._use_tensorrt else "PyTorch"
        model_size_str = "Large" if self._use_large_model else "Small"
        logger.info(
            f"Optical Flow pipeline initialized with {backend} backend, "
            f"RAFT {model_size_str} model (loads on first use)"
        )
        if not self._use_tensorrt and use_tensorrt_config:
            logger.warning(
                "TensorRT not available, using PyTorch fallback. "
                "Install with: uv sync --group tensorrt"
            )
        logger.info(f"Initialization time: {time.time() - start:.3f}s")

    def _ensure_tensorrt_engine(self):
        """Lazily initialize the TensorRT engine.

        Exports RAFT to ONNX and compiles to TRT on first call.

        Returns:
            Initialized TensorRTEngine

        Raises:
            RuntimeError: If ONNX export or TensorRT engine build fails
        """
        if self._trt_engine is not None:
            return self._trt_engine

        from .engine import (
            TensorRTEngine,
            build_tensorrt_engine,
            export_raft_to_onnx,
            get_gpu_name,
        )

        models_dir = get_models_dir()
        gpu_name = get_gpu_name()

        # Get paths (include model name for separate small/large engines)
        onnx_path = get_onnx_path(
            models_dir, self._height, self._width, self._model_name
        )
        engine_path = get_engine_path(
            models_dir, self._height, self._width, gpu_name, self._model_name
        )

        # Export RAFT to ONNX if needed
        if not onnx_path.exists():
            model_size = "Large" if self._use_large_model else "Small"
            logger.info(f"Exporting RAFT {model_size} model to ONNX...")
            success = export_raft_to_onnx(
                onnx_path=onnx_path,
                height=self._height,
                width=self._width,
                device=str(self.device),
                use_large_model=self._use_large_model,
            )
            if not success:
                raise RuntimeError("Failed to export RAFT to ONNX")

        # Build TensorRT engine if needed
        if not engine_path.exists():
            logger.info("Building TensorRT engine...")
            success = build_tensorrt_engine(
                onnx_path=onnx_path,
                engine_path=engine_path,
                min_height=self._height,
                min_width=self._width,
                max_height=self._height,
                max_width=self._width,
                fp16=self._fp16,
            )
            if not success:
                raise RuntimeError("Failed to build TensorRT engine")

        # Load and activate engine
        logger.info(f"Loading TensorRT engine: {engine_path}")
        self._trt_engine = TensorRTEngine(engine_path)
        self._trt_engine.load()
        self._trt_engine.activate()

        # Allocate buffers with input shape for dynamic shapes
        input_shape = (1, 3, self._height, self._width)
        self._trt_engine.allocate_buffers(
            device=str(self.device), input_shape=input_shape
        )

        logger.info(
            f"Optical Flow TensorRT engine initialized: {self._height}x{self._width}"
        )
        return self._trt_engine

    def _ensure_pytorch_model(self):
        """Lazily initialize the PyTorch RAFT model.

        Returns:
            Initialized RAFT model
        """
        if self._pytorch_model is not None:
            return self._pytorch_model

        from .engine import load_raft_model

        model_size = "Large" if self._use_large_model else "Small"
        logger.info(f"Loading PyTorch RAFT {model_size} model...")
        start = time.time()

        self._pytorch_model, _ = load_raft_model(
            self._use_large_model, device=str(self.device)
        )

        logger.info(
            f"Loaded PyTorch RAFT {model_size} model in {time.time() - start:.3f}s"
        )
        return self._pytorch_model

    @staticmethod
    def _pad_to_8(tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Pad NCHW tensor so H and W are multiples of 8 (RAFT requirement).

        Returns:
            Tuple of (padded tensor, original height, original width)
        """
        _, _, h, w = tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, [0, pad_w, 0, pad_h])
        return tensor, h, w

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

    def _compute_optical_flow_tensorrt(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Compute optical flow between two preprocessed frames using TensorRT.

        Args:
            frame1: First frame tensor (CHW format, [0,255], padded)
            frame2: Second frame tensor (CHW format, [0,255], padded)
            orig_h: Original height before padding
            orig_w: Original width before padding

        Returns:
            Optical flow tensor (2HW format)
        """
        engine = self._ensure_tensorrt_engine()

        # Run TensorRT inference (uses engine's dedicated stream)
        feed_dict = {"frame1": frame1.unsqueeze(0), "frame2": frame2.unsqueeze(0)}
        result = engine.infer(feed_dict)
        flow = result["flow"][0]  # Remove batch dimension

        # Remove padding
        return flow[:, :orig_h, :orig_w]

    def _compute_optical_flow_pytorch(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """Compute optical flow between two preprocessed frames using PyTorch RAFT.

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
        if self._use_tensorrt:
            return self._compute_optical_flow_tensorrt(frame1, frame2, orig_h, orig_w)
        else:
            return self._compute_optical_flow_pytorch(frame1, frame2, orig_h, orig_w)

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
