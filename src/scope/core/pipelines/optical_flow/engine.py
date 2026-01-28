"""TensorRT engine wrapper and RAFT ONNX compilation utilities."""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default resolution for optical flow computation
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


def get_gpu_name() -> str:
    """Get a sanitized GPU name for engine file naming."""
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    # Sanitize for use in filenames
    return name.lower().replace(" ", "_").replace("-", "_")


def load_raft_model(use_large_model: bool, device: str = "cuda"):
    """Load RAFT model from torchvision.

    Args:
        use_large_model: If True, load RAFT Large. Otherwise load RAFT Small.
        device: Device to load model on.

    Returns:
        Tuple of (model, weights) where weights can be used for transforms.
    """
    from torchvision.models.optical_flow import (
        Raft_Large_Weights,
        Raft_Small_Weights,
        raft_large,
        raft_small,
    )

    if use_large_model:
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True)

    model = model.to(device=device)
    model.eval()
    return model, weights


def apply_raft_transforms(
    weights, frame1: torch.Tensor, frame2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RAFT preprocessing transforms to frame pair.

    Args:
        weights: RAFT weights object (Raft_Small_Weights or Raft_Large_Weights)
        frame1: First frame tensor (BCHW format)
        frame2: Second frame tensor (BCHW format)

    Returns:
        Tuple of transformed (frame1, frame2)
    """
    if hasattr(weights, "transforms") and weights.transforms is not None:
        transforms = weights.transforms()
        return transforms(frame1, frame2)
    return frame1, frame2


class _RAFTWrapper(torch.nn.Module):
    """Wrapper to make RAFT return only the final flow prediction.

    RAFT returns a list of flow predictions (one per iteration).
    TensorRT doesn't support SequenceConstruct, so we wrap the model
    to return only the final (most refined) prediction.
    """

    def __init__(self, raft_model: torch.nn.Module):
        super().__init__()
        self.raft = raft_model

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        # RAFT returns list of predictions, take the last one
        flow_predictions = self.raft(frame1, frame2)
        return flow_predictions[-1]


def export_raft_to_onnx(
    onnx_path: Path,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    device: str = "cuda",
    use_large_model: bool = False,
) -> bool:
    """Export RAFT model from torchvision to ONNX format.

    Args:
        onnx_path: Path to save the ONNX model
        height: Input height for the model
        width: Input width for the model
        device: Device to use for export
        use_large_model: If True, export RAFT Large instead of RAFT Small

    Returns:
        True if successful, False otherwise
    """
    try:
        import torchvision.models.optical_flow  # noqa: F401
    except ImportError:
        logger.error("torchvision is required but not installed")
        return False

    model_size = "Large" if use_large_model else "Small"
    logger.info(f"Exporting RAFT {model_size} model to ONNX: {onnx_path}")
    logger.info(f"Resolution: {height}x{width}")

    try:
        # Load RAFT model using helper
        logger.info(f"Loading RAFT {model_size} model from torchvision...")
        raft_model, weights = load_raft_model(use_large_model, device)

        # Wrap model to return only final prediction (avoids SequenceConstruct)
        wrapped_model = _RAFTWrapper(raft_model)
        wrapped_model.eval()

        # Create dummy inputs at target resolution
        dummy_frame1 = torch.randn(1, 3, height, width).to(device)
        dummy_frame2 = torch.randn(1, 3, height, width).to(device)

        # Apply RAFT preprocessing transforms
        dummy_frame1, dummy_frame2 = apply_raft_transforms(
            weights, dummy_frame1, dummy_frame2
        )

        # Dynamic axes for batch, height, width
        dynamic_axes = {
            "frame1": {0: "batch_size", 2: "height", 3: "width"},
            "frame2": {0: "batch_size", 2: "height", 3: "width"},
            "flow": {0: "batch_size", 2: "height", 3: "width"},
        }

        logger.info("Exporting to ONNX...")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                wrapped_model,
                (dummy_frame1, dummy_frame2),
                str(onnx_path),
                verbose=False,
                input_names=["frame1", "frame2"],
                output_names=["flow"],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
                dynamo=False,  # Use legacy exporter (torch.export doesn't support cudnn_grid_sampler)
            )

        del wrapped_model
        del raft_model
        torch.cuda.empty_cache()

        logger.info(f"Successfully exported ONNX model to {onnx_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")
        import traceback

        traceback.print_exc()
        return False


def build_tensorrt_engine(
    onnx_path: Path,
    engine_path: Path,
    min_height: int = DEFAULT_HEIGHT,
    min_width: int = DEFAULT_WIDTH,
    max_height: int = DEFAULT_HEIGHT,
    max_width: int = DEFAULT_WIDTH,
    fp16: bool = True,
    workspace_size_gb: int = 4,
) -> bool:
    """Build TensorRT engine from ONNX model with optimization profiles.

    Args:
        onnx_path: Path to the ONNX model
        engine_path: Path to save the TensorRT engine
        min_height: Minimum input height for optimization
        min_width: Minimum input width for optimization
        max_height: Maximum input height for optimization
        max_width: Maximum input width for optimization
        fp16: Enable FP16 precision mode
        workspace_size_gb: Maximum workspace size in GB

    Returns:
        True if successful, False otherwise
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error("TensorRT is required but not installed")
        return False

    if not onnx_path.exists():
        logger.error(f"ONNX model not found: {onnx_path}")
        return False

    logger.info(f"Building TensorRT engine from ONNX model: {onnx_path}")
    logger.info(f"Output path: {engine_path}")
    logger.info(
        f"Resolution range: {min_height}x{min_width} - {max_height}x{max_width}"
    )
    logger.info(f"FP16 mode: {fp16}")
    logger.info("This may take several minutes...")

    try:
        trt_logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        logger.info("Parsing ONNX model...")
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(f"Parser error: {parser.get_error(error)}")
                return False

        logger.info("Configuring TensorRT builder...")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1 << 30)
        )

        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")

        # Calculate optimal resolution (middle point)
        opt_height = (min_height + max_height) // 2
        opt_width = (min_width + max_width) // 2

        # Create optimization profile for dynamic shapes
        profile = builder.create_optimization_profile()
        min_shape = (1, 3, min_height, min_width)
        opt_shape = (1, 3, opt_height, opt_width)
        max_shape = (1, 3, max_height, max_width)

        profile.set_shape("frame1", min_shape, opt_shape, max_shape)
        profile.set_shape("frame2", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        logger.info("Building TensorRT engine... (this will take a while)")
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return False

        logger.info(f"Saving engine to {engine_path}")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine)

        logger.info(f"Successfully built TensorRT engine: {engine_path}")
        logger.info(f"Engine size: {engine_path.stat().st_size / (1024 * 1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback

        traceback.print_exc()
        return False


class TensorRTEngine:
    """TensorRT engine wrapper for RAFT optical flow inference.

    Optimized for fixed-shape inference with minimal per-call overhead.
    """

    def __init__(self, engine_path: str | Path):
        """Initialize the engine wrapper.

        Args:
            engine_path: Path to the TensorRT engine file
        """
        self.engine_path = (
            Path(engine_path) if isinstance(engine_path, str) else engine_path
        )
        self.engine = None
        self.context = None
        self.tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._input_shape: tuple | None = None
        self._output_tensor: torch.Tensor | None = None
        self._device = "cuda"

    def load(self):
        """Load TensorRT engine from file."""
        try:
            import tensorrt as trt  # noqa: F401
            from polygraphy.backend.common import bytes_from_path
            from polygraphy.backend.trt import engine_from_bytes
        except ImportError as e:
            raise ImportError(
                "TensorRT and polygraphy are required. "
                "Install with: uv sync --group tensorrt"
            ) from e

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")

        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(str(self.engine_path)))

    def activate(self):
        """Create execution context."""
        if self.engine is None:
            raise RuntimeError("Engine must be loaded before activation")

        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, device: str = "cuda", input_shape: tuple | None = None):
        """Allocate input/output buffers for a specific input shape.

        Args:
            device: Device to allocate tensors on
            input_shape: Shape for input tensors (B, C, H, W). Required for dynamic shapes.
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("TensorRT is required") from e

        if self.context is None:
            raise RuntimeError("Context must be activated before buffer allocation")

        self._device = device
        self._input_shape = input_shape

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # For dynamic shapes, use provided input_shape
                if input_shape is not None and any(dim == -1 for dim in shape):
                    shape = input_shape
                self.context.set_input_shape(name, shape)
                # Update shape after setting it
                shape = self.context.get_tensor_shape(name)
            else:
                # For output tensors, get shape after input shapes are set
                shape = self.context.get_tensor_shape(name)

            # Verify shape has no dynamic dimensions
            if any(dim == -1 for dim in shape):
                raise RuntimeError(
                    f"Tensor '{name}' still has dynamic dimensions {shape}. "
                    f"Please provide input_shape parameter to allocate_buffers()."
                )

            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[name] = tensor

            # Cache output tensor reference for fast access
            if name == "flow":
                self._output_tensor = tensor

    def infer(
        self, feed_dict: dict[str, torch.Tensor], stream: int | None = None
    ) -> OrderedDict[str, torch.Tensor]:
        """Run inference with optional stream parameter.

        Optimized for fixed input shapes (skips reallocation checks when possible).

        Args:
            feed_dict: Dictionary mapping input names to tensors
                       For RAFT: {"frame1": tensor, "frame2": tensor}
            stream: Optional CUDA stream handle. Uses current stream if None.

        Returns:
            OrderedDict of output tensors (contains "flow" key)
        """
        # Use provided stream or default CUDA stream
        # Note: TensorRT may log a warning about default stream sync, but
        # benchmarks show this is actually faster than dedicated stream + sync
        use_stream = (
            stream if stream is not None else torch.cuda.current_stream().cuda_stream
        )

        # Fast path: check if input shape matches (common case)
        first_input = next(iter(feed_dict.values()))
        if first_input.shape != self._input_shape:
            self._reallocate_buffers(feed_dict)

        # Set input tensor addresses directly (zero-copy where possible)
        for name, buf in feed_dict.items():
            self.context.set_tensor_address(name, buf.data_ptr())

        # Set output tensor address
        self.context.set_tensor_address("flow", self._output_tensor.data_ptr())

        # Execute inference
        success = self.context.execute_async_v3(use_stream)
        if not success:
            raise ValueError("TensorRT inference failed")

        return self.tensors

    def _reallocate_buffers(self, feed_dict: dict[str, torch.Tensor]):
        """Reallocate buffers when input shape changes."""
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("TensorRT is required") from e

        # Update input shapes
        for name, buf in feed_dict.items():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, buf.shape)

        # Update cached input shape
        self._input_shape = next(iter(feed_dict.values())).shape

        # Reallocate output tensor with new shape
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                tensor = torch.empty(
                    tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
                ).to(device=self._device)
                self.tensors[name] = tensor

                if name == "flow":
                    self._output_tensor = tensor

    def __del__(self):
        """Cleanup resources."""
        self.tensors.clear()
        self._output_tensor = None
        self.context = None
        self.engine = None
