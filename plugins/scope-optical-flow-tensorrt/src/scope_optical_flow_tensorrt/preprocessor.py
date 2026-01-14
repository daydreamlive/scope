"""Main optical flow TensorRT preprocessor for VACE conditioning.

Ported from StreamDiffusion's temporal_net_tensorrt.py
"""

import logging

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.utils import flow_to_image

from .download import get_engine_path, get_models_dir, get_onnx_path
from .engine import (
    TensorRTEngine,
    build_tensorrt_engine,
    export_raft_to_onnx,
    get_gpu_name,
)

logger = logging.getLogger(__name__)


class OpticalFlowTensorRTPreprocessor:
    """TensorRT-accelerated optical flow preprocessor for VACE conditioning.

    Ported from StreamDiffusion's TemporalNetTensorRTPreprocessor.

    This preprocessor computes optical flow between consecutive frames using
    RAFT (Recurrent All-Pairs Field Transforms) accelerated by TensorRT,
    and converts it to RGB visualization for VACE/ControlNet conditioning.

    The TensorRT engine is lazily initialized on first use, automatically
    exporting RAFT from torchvision and compiling it to a GPU-specific engine.

    Example:
        ```python
        preprocessor = OpticalFlowTensorRTPreprocessor()

        # Process video frames for VACE
        # Input: [B, C, F, H, W] in [0, 1] range
        flow_maps = preprocessor(input_frames)
        # Output: [1, C, F, H, W] in [-1, 1] range
        ```
    """

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: str = "cuda",
        fp16: bool = True,
        flow_strength: float = 1.0,
    ):
        """Initialize the optical flow preprocessor.

        Args:
            height: Height for optical flow computation
            width: Width for optical flow computation
            device: Target device for inference ("cuda")
            fp16: Whether to use FP16 precision for TRT compilation
            flow_strength: Strength multiplier for flow visualization (1.0 = normal)
        """
        self._height = height
        self._width = width
        self._device = device
        self._fp16 = fp16
        self._flow_strength = max(0.0, min(2.0, flow_strength))

        self._trt_engine: TensorRTEngine | None = None
        self._is_cuda_available = torch.cuda.is_available()

        # Previous frame tracking for flow computation
        self._prev_input: torch.Tensor | None = None
        self._first_frame = True

        # Grid cache for warping (if needed)
        self._grid_cache: dict = {}

    def _ensure_engine(self) -> TensorRTEngine:
        """Lazily initialize the TensorRT engine.

        Exports RAFT to ONNX and compiles to TRT on first call.

        Returns:
            Initialized TensorRTEngine
        """
        if self._trt_engine is not None:
            return self._trt_engine

        models_dir = get_models_dir()
        gpu_name = get_gpu_name()

        # Get paths
        onnx_path = get_onnx_path(models_dir, self._height, self._width)
        engine_path = get_engine_path(models_dir, self._height, self._width, gpu_name)

        # Export RAFT to ONNX if needed
        if not onnx_path.exists():
            logger.info("Exporting RAFT model to ONNX...")
            success = export_raft_to_onnx(
                onnx_path=onnx_path,
                height=self._height,
                width=self._width,
                device=self._device,
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
        self._trt_engine.allocate_buffers(device=self._device, input_shape=input_shape)

        logger.info(
            f"Optical Flow TensorRT engine initialized: {self._height}x{self._width}"
        )
        return self._trt_engine

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
        engine = self._ensure_engine()

        # Prepare inputs for TensorRT
        frame1_batch = frame1.unsqueeze(0)
        frame2_batch = frame2.unsqueeze(0)

        # Apply RAFT preprocessing transforms
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, "transforms") and weights.transforms is not None:
            transforms = weights.transforms()
            frame1_batch, frame2_batch = transforms(frame1_batch, frame2_batch)

        # Run TensorRT inference
        feed_dict = {"frame1": frame1_batch, "frame2": frame2_batch}

        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = engine.infer(feed_dict, cuda_stream)
        flow = result["flow"][0]  # Remove batch dimension

        return flow

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
        # Convert to float32 for TensorRT processing
        prev_tensor = prev_input_tensor.to(device=self._device, dtype=torch.float32)
        current_tensor = current_input_tensor.to(
            device=self._device, dtype=torch.float32
        )

        # Resize for flow computation if needed
        if (
            current_tensor.shape[-1] != self._width
            or current_tensor.shape[-2] != self._height
        ):
            prev_resized = F.interpolate(
                prev_tensor.unsqueeze(0),
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            current_resized = F.interpolate(
                current_tensor.unsqueeze(0),
                size=(self._height, self._width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            prev_resized = prev_tensor
            current_resized = current_tensor

        # Compute optical flow using TensorRT: prev_input -> current_input
        flow = self._compute_optical_flow_tensorrt(prev_resized, current_resized)

        # Apply flow strength scaling
        if self._flow_strength != 1.0:
            flow = flow * self._flow_strength

        # Convert flow to RGB visualization using torchvision's flow_to_image
        # flow_to_image expects (2, H, W) and returns (3, H, W) in range [0, 255]
        flow_rgb = flow_to_image(flow)  # Returns uint8 tensor [0, 255]

        # Convert to float [0, 1] range
        flow_rgb = flow_rgb.float() / 255.0

        return flow_rgb

    def reset(self):
        """Reset the preprocessor state.

        Call this between different video sequences to clear the
        previous frame buffer.
        """
        self._first_frame = True
        self._prev_input = None
        self._grid_cache.clear()
        torch.cuda.empty_cache()

    def __call__(
        self,
        frames: torch.Tensor,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        """Process input frames to produce optical flow maps for VACE conditioning.

        Args:
            frames: Input video frames. Supported formats:
                - [B, C, F, H, W]: Batch of videos (B batches, F frames each)
                - [C, F, H, W]: Single video (F frames)
                - [F, H, W, C]: Single video in FHWC format (will be converted)
                All inputs should be in [0, 1] float range.
            target_height: Output height. If None, uses flow computation height.
            target_width: Output width. If None, uses flow computation width.

        Returns:
            Flow maps in VACE format: [1, C, F, H, W] float in [-1, 1] range,
            with optical flow rendered as RGB visualization.
        """
        # Handle different input formats
        if frames.dim() == 4:
            if frames.shape[-1] == 3:
                # [F, H, W, C] -> [C, F, H, W]
                frames = frames.permute(3, 0, 1, 2)
            # [C, F, H, W] -> [1, C, F, H, W]
            frames = frames.unsqueeze(0)

        batch_size, channels, num_frames, height, width = frames.shape

        # Default target size to flow resolution
        if target_height is None:
            target_height = self._height
        if target_width is None:
            target_width = self._width

        # Process each batch
        all_flow_frames = []

        for b in range(batch_size):
            # Reset previous frame for each batch
            self.reset()

            batch_flows = []
            for f in range(num_frames):
                frame = frames[b, :, f]  # [C, H, W]

                # Normalize input tensor if needed
                input_tensor = frame
                if input_tensor.max() > 1.0:
                    input_tensor = input_tensor / 255.0

                # Ensure on GPU
                if self._is_cuda_available and not input_tensor.is_cuda:
                    input_tensor = input_tensor.cuda()

                if self._prev_input is not None and not self._first_frame:
                    try:
                        # Compute optical flow between prev_input -> current_input
                        flow_rgb = self._compute_flow_to_rgb_tensor(
                            self._prev_input, input_tensor
                        )
                    except Exception as e:
                        logger.error(f"TensorRT optical flow failed: {e}")
                        # Fallback: zero flow
                        flow_rgb = torch.zeros(
                            3,
                            self._height,
                            self._width,
                            device=self._device,
                            dtype=torch.float32,
                        )
                else:
                    # First frame: no previous frame, output zero flow
                    self._first_frame = False
                    flow_rgb = torch.zeros(
                        3,
                        self._height,
                        self._width,
                        device=self._device,
                        dtype=torch.float32,
                    )

                # Resize to target size if needed
                if (
                    flow_rgb.shape[-2] != target_height
                    or flow_rgb.shape[-1] != target_width
                ):
                    flow_rgb = F.interpolate(
                        flow_rgb.unsqueeze(0),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)

                # Store current input as previous for next frame
                self._prev_input = input_tensor.clone()

                batch_flows.append(flow_rgb)

            all_flow_frames.append(batch_flows)

        # Stack and convert to tensor [B, F, C, H, W]
        flow_tensor = torch.stack([torch.stack(batch) for batch in all_flow_frames])

        # [B, F, C, H, W] -> [B, C, F, H, W]
        flow_tensor = flow_tensor.permute(0, 2, 1, 3, 4)

        # Convert from [0, 1] to [-1, 1] for VACE format
        flow_tensor = flow_tensor * 2.0 - 1.0

        return flow_tensor

    def process_pil_image(self, image, prev_image=None) -> torch.Tensor:
        """Process PIL images to compute optical flow.

        Convenience method for processing individual image pairs.

        Args:
            image: Current PIL Image in RGB format
            prev_image: Previous PIL Image (if None, outputs zero flow)

        Returns:
            Flow map tensor [1, C, 1, H, W] in [-1, 1] range
        """
        import numpy as np

        # Reset state
        self.reset()

        # Convert current image
        img_array = np.array(image)
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        # [H, W, C] -> [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)

        if prev_image is not None:
            # Process previous frame first to set up state
            prev_array = np.array(prev_image)
            prev_tensor = torch.from_numpy(prev_array).float() / 255.0
            prev_tensor = prev_tensor.permute(2, 0, 1)

            # Stack as [C, F, H, W] with prev then current
            frames = torch.stack([prev_tensor, img_tensor], dim=1)
            frames = frames.unsqueeze(0)  # [1, C, F, H, W]

            result = self(frames)
            # Return only the second frame's flow
            return result[:, :, 1:2, :, :]
        else:
            # Single frame - zero flow
            frames = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
            return self(frames)

    def warmup(self):
        """Warm up the TensorRT engine.

        Call this before processing to avoid latency on first frame.
        """
        logger.info("Warming up optical flow TensorRT engine...")
        self._ensure_engine()

        # Run a dummy inference with 2 frames to test flow computation
        dummy_input = torch.zeros(1, 3, 2, self._height, self._width)
        if self._is_cuda_available:
            dummy_input = dummy_input.cuda()

        _ = self(dummy_input)
        self.reset()
        logger.info("Warmup complete")
