"""TensorRT engine runner using polygraphy.

This module provides a TensorRT engine runner that wraps the engine
and handles inference with PyTorch tensor I/O.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check for TensorRT availability
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    from polygraphy.backend.common import BytesFromPath
    from polygraphy.backend.trt import EngineFromBytes, TrtRunner

    TRT_AVAILABLE = True
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

        Args:
            **inputs: Named inputs as PyTorch tensors or numpy arrays.
        """
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                # Convert to numpy, ensuring contiguous memory
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

    def __call__(
        self,
        output_names: list[str] | None = None,
        return_torch: bool = True,
        **inputs: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Run inference on the TensorRT engine.

        Args:
            output_names: List of output names to return. If None, returns all outputs.
            return_torch: If True, return PyTorch tensors. Otherwise, return numpy arrays.
            **inputs: Named inputs as PyTorch tensors or numpy arrays.

        Returns:
            Dictionary mapping output names to tensors/arrays.
        """
        if self._runner is None:
            raise RuntimeError("TensorRT runner not initialized")

        # Prepare feed dict with prefilled values
        feed_dict = dict(self._prefilled)

        # Add current inputs
        for name, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            feed_dict[name] = np.ascontiguousarray(tensor)

        # Run inference
        outputs = self._runner.infer(feed_dict=feed_dict)

        # Filter outputs if requested
        if output_names is not None:
            outputs = {k: v for k, v in outputs.items() if k in output_names}

        # Update prefilled values with bound outputs
        for output_name, input_name in self._output_bindings.items():
            if output_name in outputs:
                self._prefilled[input_name] = outputs[output_name]

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
