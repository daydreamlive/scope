"""TensorRT engine runner using polygraphy.

This module provides a TensorRT engine runner that wraps the engine
and handles inference with PyTorch tensor I/O.

Uses CUDA device buffers (DeviceView) to avoid CPU-GPU memory transfers.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check for TensorRT availability
TRT_AVAILABLE = False
CUDA_BUFFERS_AVAILABLE = False
try:
    import tensorrt as trt
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    TRT_AVAILABLE = True

    # Check for CUDA buffer support (polygraphy >= 0.47)
    try:
        from polygraphy.cuda import DeviceView
        CUDA_BUFFERS_AVAILABLE = True
    except ImportError:
        logger.warning("polygraphy.cuda.DeviceView not available - using numpy path (slower)")
except ImportError:
    pass


class TRTRunner:
    """TensorRT engine runner with PyTorch tensor support.

    This class wraps a TensorRT engine and provides methods for inference
    with PyTorch tensors as input/output.

    Attributes:
        engine_path: Path to the TensorRT engine file.
        device: CUDA device for inference.
        runner: Polygraphy TrtRunner instance.
    """

    def __init__(
        self,
        engine_path: Path | str,
        device: torch.device | None = None,
    ):
        """Initialize TensorRT runner.

        Args:
            engine_path: Path to the TensorRT engine file.
            device: CUDA device for inference.

        Raises:
            RuntimeError: If TensorRT is not available.
            FileNotFoundError: If engine file doesn't exist.
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Install with: pip install daydream-scope[tensorrt]")

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.engine_path = engine_path
        self.device = device or torch.device("cuda")
        self._runner = None
        self._engine = None
        self._context = None

        # Pre-filled inputs (constants that don't change per frame)
        self._prefilled: dict[str, np.ndarray] = {}

        # Output bindings (for reusing memory between outputs and inputs)
        self._output_bindings: dict[str, str] = {}

        logger.info(f"Loading TensorRT engine from {engine_path}")
        self._load_engine()

        if CUDA_BUFFERS_AVAILABLE:
            logger.info("TensorRT runner using CUDA device buffers (zero-copy)")
        else:
            logger.warning("TensorRT runner using numpy path (slower - install polygraphy>=0.47 for zero-copy)")

    def _load_engine(self):
        """Load the TensorRT engine."""
        # Load engine bytes
        engine_bytes = BytesFromPath(str(self.engine_path))

        # Create engine from bytes
        self._engine = EngineFromBytes(engine_bytes)

        # Create runner (context manager handles activation)
        self._runner = TrtRunner(self._engine)
        self._runner.activate()

        logger.info("TensorRT engine loaded successfully")

    def prefill(self, **inputs: torch.Tensor | np.ndarray):
        """Pre-fill constant inputs that don't change per frame.

        These inputs will be automatically included in every inference call.
        When CUDA buffers are available, keeps tensors on GPU for zero-copy inference.

        Args:
            **inputs: Named inputs as PyTorch tensors or numpy arrays.
        """
        for name, tensor in inputs.items():
            if CUDA_BUFFERS_AVAILABLE:
                # Keep as torch tensor on GPU for zero-copy inference
                if isinstance(tensor, np.ndarray):
                    tensor = torch.from_numpy(tensor).to(self.device).contiguous()
                elif isinstance(tensor, torch.Tensor):
                    if not tensor.is_cuda:
                        tensor = tensor.to(self.device)
                    tensor = tensor.contiguous()
                self._prefilled[name] = tensor
            else:
                # Fallback to numpy for compatibility
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.detach().cpu().numpy()
                self._prefilled[name] = np.ascontiguousarray(tensor)

    def bind(self, output_to_input: dict[str, str]):
        """Bind output tensors to input tensors for memory reuse.

        This allows outputs from one inference to be used as inputs
        to the next inference without memory copies.

        Args:
            output_to_input: Mapping from output names to input names.
        """
        self._output_bindings = output_to_input.copy()

    def clear_prefill(self):
        """Clear all pre-filled inputs."""
        self._prefilled.clear()

    def _tensor_to_device_view(self, tensor: torch.Tensor):
        """Convert PyTorch tensor to polygraphy DeviceView (zero-copy).

        Args:
            tensor: PyTorch CUDA tensor (must be contiguous).

        Returns:
            DeviceView pointing to the tensor's GPU memory.
        """
        if not CUDA_BUFFERS_AVAILABLE:
            raise RuntimeError("DeviceView not available")

        # Ensure tensor is contiguous and on CUDA
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if not tensor.is_cuda:
            tensor = tensor.cuda()

        # Map PyTorch dtype to numpy dtype properly
        dtype_map = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        np_dtype = dtype_map.get(tensor.dtype, np.float32)

        # Create DeviceView from tensor's data pointer
        return DeviceView(
            ptr=tensor.data_ptr(),
            shape=tuple(tensor.shape),
            dtype=np_dtype,
        )

    def __call__(
        self,
        output_names: list[str] | None = None,
        return_torch: bool = True,
        **inputs: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Run inference on the TensorRT engine.

        Uses CUDA device buffers when available to avoid CPU-GPU memory transfers.

        Args:
            output_names: List of output names to return. If None, returns all outputs.
            return_torch: If True, return PyTorch tensors. Otherwise, return numpy arrays.
            **inputs: Named inputs as PyTorch tensors or numpy arrays.

        Returns:
            Dictionary mapping output names to tensors/arrays.
        """
        if self._runner is None:
            raise RuntimeError("TensorRT runner not initialized")

        # Use CUDA device buffers if available (zero-copy path)
        if CUDA_BUFFERS_AVAILABLE and return_torch:
            return self._call_cuda_buffers(output_names, **inputs)

        # Fallback to numpy path (has CPU-GPU copy overhead)
        return self._call_numpy(output_names, return_torch, **inputs)

    def _call_cuda_buffers(
        self,
        output_names: list[str] | None = None,
        **inputs: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Run inference using CUDA device buffers (zero-copy).

        Args:
            output_names: List of output names to return.
            **inputs: Named inputs as PyTorch tensors.

        Returns:
            Dictionary mapping output names to PyTorch tensors.
        """
        # Prepare feed dict with DeviceView wrappers
        feed_dict = {}

        # Add prefilled values (convert from numpy to DeviceView if needed)
        for name, arr in self._prefilled.items():
            if isinstance(arr, np.ndarray):
                # Convert numpy prefill to torch tensor (one-time cost)
                tensor = torch.from_numpy(arr).to(self.device).contiguous()
                self._prefilled[name] = tensor
                feed_dict[name] = self._tensor_to_device_view(tensor)
            elif isinstance(arr, torch.Tensor):
                feed_dict[name] = self._tensor_to_device_view(arr)
            else:
                # Already a DeviceView or compatible
                feed_dict[name] = arr

        # Add current inputs
        for name, tensor in inputs.items():
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).to(self.device)
            if isinstance(tensor, torch.Tensor):
                if not tensor.is_cuda:
                    tensor = tensor.to(self.device)
                feed_dict[name] = self._tensor_to_device_view(tensor.contiguous())

        # Run inference
        outputs = self._runner.infer(feed_dict=feed_dict)

        # Convert outputs to PyTorch tensors (they come back as DeviceView or numpy)
        result = {}
        for name, output in outputs.items():
            if hasattr(output, 'numpy'):
                # DeviceView - create tensor from same memory
                arr = output.numpy()
                result[name] = torch.from_numpy(arr).to(self.device)
            elif isinstance(output, np.ndarray):
                result[name] = torch.from_numpy(output).to(self.device)
            elif isinstance(output, torch.Tensor):
                result[name] = output
            else:
                result[name] = output

        # Update prefilled values with bound outputs
        for output_name, input_name in self._output_bindings.items():
            if output_name in result:
                self._prefilled[input_name] = result[output_name]

        # Filter outputs if requested
        if output_names is not None:
            result = {k: v for k, v in result.items() if k in output_names}

        return result

    def _call_numpy(
        self,
        output_names: list[str] | None = None,
        return_torch: bool = True,
        **inputs: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Run inference using numpy arrays (has CPU-GPU copy overhead).

        Args:
            output_names: List of output names to return.
            return_torch: If True, return PyTorch tensors.
            **inputs: Named inputs as PyTorch tensors or numpy arrays.

        Returns:
            Dictionary mapping output names to tensors/arrays.
        """
        # Prepare feed dict with prefilled values
        feed_dict = {}

        # Add prefilled numpy values
        for name, arr in self._prefilled.items():
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            feed_dict[name] = np.ascontiguousarray(arr)

        # Add current inputs
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            feed_dict[name] = np.ascontiguousarray(tensor)

        # Run inference
        outputs = self._runner.infer(feed_dict=feed_dict)

        # Update prefilled values with bound outputs BEFORE filtering
        # This ensures recurrent states (latents->sample) are properly updated
        for output_name, input_name in self._output_bindings.items():
            if output_name in outputs:
                self._prefilled[input_name] = outputs[output_name]

        # Filter outputs if requested (after bindings are applied)
        if output_names is not None:
            outputs = {k: v for k, v in outputs.items() if k in output_names}

        # Convert to PyTorch if requested
        if return_torch:
            outputs = {
                name: torch.from_numpy(arr).to(self.device)
                for name, arr in outputs.items()
            }

        return outputs

    def get_input_names(self) -> list[str]:
        """Get list of input tensor names."""
        if self._runner is None:
            return []
        return list(self._runner.get_input_metadata().keys())

    def get_output_names(self) -> list[str]:
        """Get list of output tensor names."""
        if self._runner is None:
            return []
        return list(self._runner.get_output_metadata().keys())

    def get_input_shapes(self) -> dict[str, tuple]:
        """Get input tensor shapes."""
        if self._runner is None:
            return {}
        return {name: tuple(meta.shape) for name, meta in self._runner.get_input_metadata().items()}

    def get_output_shapes(self) -> dict[str, tuple]:
        """Get output tensor shapes."""
        if self._runner is None:
            return {}
        return {name: tuple(meta.shape) for name, meta in self._runner.get_output_metadata().items()}

    def __del__(self):
        """Clean up resources."""
        if self._runner is not None:
            try:
                self._runner.deactivate()
            except Exception:
                pass

    def __repr__(self) -> str:
        """String representation."""
        inputs = self.get_input_shapes()
        outputs = self.get_output_shapes()
        return (
            f"TRTRunner(\n"
            f"  engine={self.engine_path.name},\n"
            f"  inputs={list(inputs.keys())},\n"
            f"  outputs={list(outputs.keys())}\n"
            f")"
        )
