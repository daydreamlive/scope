"""TensorRT engine wrapper and ONNX compilation utilities."""

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


def compile_onnx_to_trt(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
) -> Path:
    """Compile ONNX model to TensorRT engine using polygraphy.

    Args:
        onnx_path: Path to input ONNX model
        engine_path: Path where TRT engine will be saved
        fp16: Whether to use FP16 precision (recommended)

    Returns:
        Path to the compiled engine

    Raises:
        RuntimeError: If compilation fails
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if engine_path.exists():
        logger.info(f"TensorRT engine already exists: {engine_path}")
        return engine_path

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    try:
        import tensorrt as trt
        from polygraphy.backend.trt import (
            CreateConfig,
            EngineFromNetwork,
            NetworkFromOnnxPath,
            Profile,
            SaveEngine,
        )
    except ImportError as e:
        raise ImportError(
            "TensorRT and polygraphy are required for engine compilation. "
            "Install with: pip install tensorrt polygraphy"
        ) from e

    logger.info(f"Compiling ONNX to TensorRT: {onnx_path} -> {engine_path}")
    logger.info("This may take several minutes on first run...")

    try:
        # Build network from ONNX
        build_network = NetworkFromOnnxPath(str(onnx_path))

        # Configure build settings
        config_kwargs = {}
        if fp16:
            config_kwargs["fp16"] = True

        create_config = CreateConfig(**config_kwargs)

        # Build engine
        build_engine = EngineFromNetwork(build_network, config=create_config)

        # Save engine to file
        save_engine = SaveEngine(build_engine, str(engine_path))

        # Execute the pipeline (this triggers the actual build)
        engine = save_engine()

        # Clean up
        del engine

    except Exception as e:
        logger.error(f"TensorRT compilation failed: {e}")
        raise RuntimeError(f"TensorRT compilation failed: {e}") from e

    if not engine_path.exists():
        raise RuntimeError(f"Engine file was not created: {engine_path}")

    logger.info(f"Successfully compiled TensorRT engine: {engine_path}")
    return engine_path


class TensorRTEngine:
    """TensorRT engine wrapper for pose estimation inference.

    Handles engine loading, buffer allocation, and inference execution.
    """

    def __init__(self, engine_path: str | Path):
        """Initialize the engine wrapper.

        Args:
            engine_path: Path to the TensorRT engine file
        """
        self.engine_path = Path(engine_path)
        self.engine = None
        self.context = None
        self.tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cuda_stream = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    @property
    def input_name(self) -> str:
        """Get the primary input tensor name."""
        if not self._input_names:
            raise RuntimeError("Buffers not allocated yet")
        return self._input_names[0]

    @property
    def output_names(self) -> list[str]:
        """Get all output tensor names."""
        return self._output_names

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
        """Create execution context and cache CUDA stream."""
        if self.engine is None:
            raise RuntimeError("Engine must be loaded before activation")

        self.context = self.engine.create_execution_context()
        self._cuda_stream = torch.cuda.current_stream().cuda_stream

    def allocate_buffers(self, device: str = "cuda"):
        """Allocate input/output buffers based on engine bindings.

        Args:
            device: Target device for buffer allocation
        """
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError("TensorRT is required") from e

        if self.context is None:
            raise RuntimeError("Context must be activated before buffer allocation")

        self._input_names = []
        self._output_names = []

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                self.context.set_input_shape(name, shape)
                self._input_names.append(name)
            else:
                self._output_names.append(name)

            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[name] = tensor

            io_type = "input" if is_input else "output"
            logger.debug(
                f"Allocated {io_type} buffer '{name}': shape={shape}, dtype={dtype}"
            )

    def infer(
        self, feed_dict: dict[str, torch.Tensor], stream: int | None = None
    ) -> OrderedDict[str, torch.Tensor]:
        """Run inference with provided input tensors.

        Args:
            feed_dict: Dictionary mapping input names to tensors
            stream: Optional CUDA stream handle. Uses cached stream if None.

        Returns:
            OrderedDict of output tensors

        Raises:
            ValueError: If inference fails
        """
        if stream is None:
            stream = self._cuda_stream

        # Copy input data to pre-allocated tensors
        for name, buf in feed_dict.items():
            if name in self.tensors:
                self.tensors[name].copy_(buf)

        # Set tensor addresses for all I/O tensors
        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Execute inference asynchronously
        success = self.context.execute_async_v3(stream)
        if not success:
            raise ValueError("TensorRT inference failed")

        return self.tensors

    def __del__(self):
        """Cleanup resources."""
        self.tensors.clear()
        self.context = None
        self.engine = None
