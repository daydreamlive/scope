"""TensorRT engine wrapper and RAFT ONNX compilation utilities.

Ported from StreamDiffusion's temporal_net_tensorrt.py and compile_raft_tensorrt.py
"""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

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


def export_raft_to_onnx(
    onnx_path: Path,
    height: int = 512,
    width: int = 512,
    device: str = "cuda",
) -> bool:
    """Export RAFT model from torchvision to ONNX format.

    Ported from StreamDiffusion's compile_raft_tensorrt.py

    Args:
        onnx_path: Path to save the ONNX model
        height: Input height for the model
        width: Input width for the model
        device: Device to use for export

    Returns:
        True if successful, False otherwise
    """
    try:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
    except ImportError:
        logger.error("torchvision is required but not installed")
        return False

    logger.info(f"Exporting RAFT model to ONNX: {onnx_path}")
    logger.info(f"Resolution: {height}x{width}")

    try:
        # Load RAFT model
        logger.info("Loading RAFT Small model from torchvision...")
        raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
        raft_model = raft_model.to(device=device)
        raft_model.eval()

        # Create dummy inputs at target resolution
        dummy_frame1 = torch.randn(1, 3, height, width).to(device)
        dummy_frame2 = torch.randn(1, 3, height, width).to(device)

        # Apply RAFT preprocessing transforms
        weights = Raft_Small_Weights.DEFAULT
        if hasattr(weights, "transforms") and weights.transforms is not None:
            transforms = weights.transforms()
            dummy_frame1, dummy_frame2 = transforms(dummy_frame1, dummy_frame2)

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
                raft_model,
                (dummy_frame1, dummy_frame2),
                str(onnx_path),
                verbose=False,
                input_names=["frame1", "frame2"],
                output_names=["flow"],
                opset_version=17,
                export_params=True,
                dynamic_axes=dynamic_axes,
            )

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
    min_height: int = 512,
    min_width: int = 512,
    max_height: int = 512,
    max_width: int = 512,
    fp16: bool = True,
    workspace_size_gb: int = 4,
) -> bool:
    """Build TensorRT engine from ONNX model with optimization profiles.

    Ported from StreamDiffusion's compile_raft_tensorrt.py

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
        logger.info(f"Engine size: {engine_path.stat().st_size / (1024*1024):.2f} MB")

        return True

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        import traceback

        traceback.print_exc()
        return False


class TensorRTEngine:
    """TensorRT engine wrapper for RAFT optical flow inference.

    Ported from StreamDiffusion's temporal_net_tensorrt.py.
    Handles dynamic shapes and buffer reallocation.
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
        self._cuda_stream = None

    def load(self):
        """Load TensorRT engine from file."""
        try:
            import tensorrt as trt
            from polygraphy.backend.common import bytes_from_path
            from polygraphy.backend.trt import engine_from_bytes
        except ImportError as e:
            raise ImportError(
                "TensorRT and polygraphy are required. "
                "Install with: pip install tensorrt polygraphy"
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
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device: str = "cuda", input_shape: tuple | None = None):
        """Allocate input/output buffers.

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

    def infer(
        self, feed_dict: dict[str, torch.Tensor], stream: int | None = None
    ) -> OrderedDict[str, torch.Tensor]:
        """Run inference with optional stream parameter.

        Handles dynamic shape reallocation if input shapes change.

        Args:
            feed_dict: Dictionary mapping input names to tensors
                       For RAFT: {"frame1": tensor, "frame2": tensor}
            stream: Optional CUDA stream handle. Uses cached stream if None.

        Returns:
            OrderedDict of output tensors (contains "flow" key)
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("TensorRT is required") from e

        if stream is None:
            stream = self._cuda_stream

        # Check if we need to update tensor shapes for dynamic dimensions
        need_realloc = False
        for name, buf in feed_dict.items():
            if name in self.tensors:
                if self.tensors[name].shape != buf.shape:
                    need_realloc = True
                    break

        # Reallocate buffers if input shape changed
        if need_realloc:
            # Update input shapes
            for name, buf in feed_dict.items():
                try:
                    if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        self.context.set_input_shape(name, buf.shape)
                except Exception:
                    pass

            # Reallocate all tensors with new shapes
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)
                shape = self.context.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                tensor = torch.empty(
                    tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
                ).to(device=self.tensors[name].device)
                self.tensors[name] = tensor

        # Copy input data to tensors
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)

        # Set tensor addresses
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Execute inference
        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("TensorRT inference failed")

        return self.tensors

    def __del__(self):
        """Cleanup resources."""
        self.tensors.clear()
        self.context = None
        self.engine = None
