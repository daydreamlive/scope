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
        self._flow_strength = 1.0

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

        # Previous frame tracking for flow computation
        self._prev_input: torch.Tensor | None = None
        self._first_frame = True

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

    def _compute_optical_flow_tensorrt(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> torch.Tensor:
        """Compute optical flow between two frames using TensorRT-accelerated RAFT.

        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])

        Returns:
            Optical flow tensor (2HW format)
        """
        engine = self._ensure_tensorrt_engine()

        # Convert to [0, 255] float and add batch dim (matching VACE reference)
        frame1_batch = (frame1 * 255.0).unsqueeze(0)
        frame2_batch = (frame2 * 255.0).unsqueeze(0)

        # Pad to multiple of 8
        frame1_batch, h, w = self._pad_to_8(frame1_batch)
        frame2_batch, _, _ = self._pad_to_8(frame2_batch)

        # Run TensorRT inference
        feed_dict = {"frame1": frame1_batch, "frame2": frame2_batch}

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = engine.infer(feed_dict, cuda_stream)
        flow = result["flow"][0]  # Remove batch dimension

        # Remove padding
        flow = flow[:, :h, :w]

        return flow

    def _compute_optical_flow_pytorch(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> torch.Tensor:
        """Compute optical flow between two frames using PyTorch RAFT.

        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])

        Returns:
            Optical flow tensor (2HW format)
        """
        model = self._ensure_pytorch_model()

        # Convert to [0, 255] float and add batch dim (matching VACE reference)
        frame1_batch = (frame1 * 255.0).unsqueeze(0)
        frame2_batch = (frame2 * 255.0).unsqueeze(0)

        # Pad to multiple of 8 (RAFT requirement)
        frame1_batch, h, w = self._pad_to_8(frame1_batch)
        frame2_batch, _, _ = self._pad_to_8(frame2_batch)

        # Run PyTorch inference
        with torch.no_grad():
            flow_predictions = model(frame1_batch, frame2_batch)
            flow = flow_predictions[-1][0]  # Last prediction, remove batch dim

        # Remove padding
        flow = flow[:, :h, :w]

        return flow

    def _compute_optical_flow(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> torch.Tensor:
        """Compute optical flow between two frames using the configured backend.

        Args:
            frame1: First frame tensor (CHW format, [0,1])
            frame2: Second frame tensor (CHW format, [0,1])

        Returns:
            Optical flow tensor (2HW format)
        """
        if self._use_tensorrt:
            return self._compute_optical_flow_tensorrt(frame1, frame2)
        else:
            return self._compute_optical_flow_pytorch(frame1, frame2)

    def _compute_flow_to_rgb_tensor(
        self, prev_input_tensor: torch.Tensor, current_input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Compute optical flow and convert to RGB visualization.

        Args:
            prev_input_tensor: Previous input frame tensor (CHW format, [0,1]) on GPU
            current_input_tensor: Current input frame tensor (CHW format, [0,1]) on GPU

        Returns:
            Flow visualization as RGB tensor (CHW format, [0,1]) on GPU
        """
        # Convert to float32 for processing
        prev_tensor = prev_input_tensor.to(device=self.device, dtype=torch.float32)
        current_tensor = current_input_tensor.to(
            device=self.device, dtype=torch.float32
        )

        # Resize for flow computation if needed
        prev_resized = self._resize_flow_to_target(
            prev_tensor, self._height, self._width
        )
        current_resized = self._resize_flow_to_target(
            current_tensor, self._height, self._width
        )

        # Compute optical flow: prev_input -> current_input
        flow = self._compute_optical_flow(prev_resized, current_resized)

        # Apply flow strength scaling
        if self._flow_strength != 1.0:
            flow = flow * self._flow_strength

        # Convert flow to RGB visualization using torchvision's flow_to_image
        # flow_to_image expects (2, H, W) and returns (3, H, W) in range [0, 255]
        flow_rgb = flow_to_image(flow)  # Returns uint8 tensor [0, 255]

        # Convert to float [0, 1] range
        return flow_rgb.float() / 255.0

    def reset(self):
        """Reset the pipeline state.

        Call this between different video sequences to clear the
        previous frame buffer.
        """
        self._first_frame = True
        self._prev_input = None

    def prepare(self, **kwargs) -> Requirements:
        """Return pipeline requirements.

        Returns:
            Requirements specifying input_size needed for temporal consistency
        """
        return Requirements(input_size=4)

    def _normalize_frame(
        self, frame, target_shape: tuple | None
    ) -> tuple[torch.Tensor, tuple]:
        """Normalize a single frame to consistent format.

        Args:
            frame: Input frame (tensor or numpy array)
            target_shape: Target shape for consistency, or None to use this frame's shape

        Returns:
            Tuple of (normalized frame tensor in HWC [0,1], target_shape)
        """
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame
        else:
            frame_tensor = torch.from_numpy(frame)

        # Squeeze T dimension: (1, H, W, C) -> (H, W, C)
        frame_tensor = frame_tensor.squeeze(0)

        # Use first frame's shape as target for consistency
        if target_shape is None:
            target_shape = frame_tensor.shape
        elif frame_tensor.shape != target_shape:
            # Resize frame to match target shape
            frame_chw = frame_tensor.permute(2, 0, 1).unsqueeze(0).float()
            frame_chw = F.interpolate(
                frame_chw,
                size=(target_shape[0], target_shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            frame_tensor = frame_chw.squeeze(0).permute(1, 2, 0)

        # Normalize to [0, 1] if needed
        if frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor.float() / 255.0
        else:
            frame_tensor = frame_tensor.float()

        return frame_tensor, target_shape

    def _resize_flow_to_target(
        self, flow: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """Resize flow tensor to target dimensions if needed.

        Args:
            flow: Flow tensor in CHW format
            h: Target height
            w: Target width

        Returns:
            Resized flow tensor
        """
        if flow.shape[-2] != h or flow.shape[-1] != w:
            flow = F.interpolate(
                flow.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return flow

    def _create_zero_flow(self, h: int, w: int) -> torch.Tensor:
        """Create a zero flow tensor.

        Args:
            h: Height
            w: Width

        Returns:
            Zero flow tensor in CHW format
        """
        return torch.zeros(
            3,
            h,
            w,
            device=self.device,
            dtype=torch.float32,
        )

    def __call__(self, **kwargs) -> torch.Tensor:
        """Process video frames and return optical flow visualizations.

        Args:
            video: Input video frames as list of tensors (THWC format, [0, 255] range)

        Returns:
            Flow maps as tensor in THWC format with values in [0, 1] range,
            rendered as RGB visualizations.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for OpticalFlowPipeline")

        # Reset state for each new call
        self.reset()

        # Normalize all frames
        frames = []
        target_shape = None
        for frame in video:
            frame_tensor, target_shape = self._normalize_frame(frame, target_shape)
            frames.append(frame_tensor)

        num_frames = len(frames)
        h, w = target_shape[0], target_shape[1]

        # Process each frame to compute optical flow
        flow_frames = []
        first_computed_flow = None

        for i in range(num_frames):
            # HWC -> CHW
            frame_chw = frames[i].permute(2, 0, 1)

            # Ensure on GPU
            if self._is_cuda_available and not frame_chw.is_cuda:
                frame_chw = frame_chw.to(self.device)

            if self._prev_input is not None and not self._first_frame:
                try:
                    # Compute optical flow between prev_input -> current_input
                    flow_rgb = self._compute_flow_to_rgb_tensor(
                        self._prev_input, frame_chw
                    )
                    # Store the first computed flow for duplicating to frame 0
                    # (matches VACE's FlowVisAnnotator behavior)
                    if first_computed_flow is None:
                        first_computed_flow = flow_rgb.clone()
                except Exception as e:
                    logger.error(f"Optical flow computation failed: {e}")
                    # Fallback: zero flow
                    flow_rgb = self._create_zero_flow(self._height, self._width)
            else:
                # First frame: placeholder, will be replaced with first computed flow
                self._first_frame = False
                flow_rgb = None  # Placeholder

            # Resize to target size if needed
            if flow_rgb is not None:
                flow_rgb = self._resize_flow_to_target(flow_rgb, h, w)

            # Store current input as previous for next frame
            self._prev_input = frame_chw.clone()

            flow_frames.append(flow_rgb)

        # Replace first frame's placeholder with duplicated first computed flow
        # This matches VACE's behavior: flow_up_vis_list[:1] + flow_up_vis_list
        if first_computed_flow is not None:
            flow_frames[0] = self._resize_flow_to_target(first_computed_flow, h, w)
        else:
            # Single frame or no valid flow computed - use zero flow
            flow_frames[0] = self._create_zero_flow(h, w)

        # Stack flows: [T, C, H, W]
        flow_tensor = torch.stack(flow_frames, dim=0)

        # Convert from CHW to HWC: [T, C, H, W] -> [T, H, W, C]
        flow_tensor = flow_tensor.permute(0, 2, 3, 1)

        # Output is already in [0, 1] range in THWC format
        return flow_tensor
