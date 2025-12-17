"""TensorRT engine runner using pycuda for direct CUDA memory management.

Based on PersonaLive official implementation:
PersonaLive/src/modeling/engine_model.py

This provides significantly better performance than polygraphy's TrtRunner
by using direct CUDA memory allocation and device-to-device copies.
"""

import logging
import os
import traceback
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Check for TensorRT and pycuda availability
PYCUDA_TRT_AVAILABLE = False
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - Required for pycuda initialization

    PYCUDA_TRT_AVAILABLE = True
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except ImportError as e:
    logger.debug(f"pycuda/tensorrt not available: {e}")


def _get_engine(engine_file_path: str):
    """Load TensorRT engine from file."""
    if os.path.exists(engine_file_path):
        logger.info(f"Loading engine from file {engine_file_path}...")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        raise FileNotFoundError(f"Engine file not found: {engine_file_path}")


def _numpy_to_torch_dtype(np_dtype):
    """Convert numpy dtype to torch dtype."""
    mapping = {
        np.float32: torch.float,
        np.float64: torch.double,
        np.float16: torch.half,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int16: torch.int16,
        np.int8: torch.int8,
        np.uint8: torch.uint8,
        np.bool_: torch.bool,
    }
    return mapping.get(np_dtype, torch.float)


class EngineModel:
    """TensorRT engine runner with pycuda for direct CUDA memory management.

    This provides significantly better performance than polygraphy by:
    1. Pre-allocating CUDA device memory for all inputs/outputs
    2. Using device-to-device copies (memcpy_dtod_async) when possible
    3. Using async TRT execution with CUDA streams
    4. True zero-copy binding for recurrent state

    Based on PersonaLive official implementation.
    """

    def __init__(
        self,
        engine_file_path: str | Path,
        device_int: int = 0,
    ):
        """Initialize TensorRT engine with pycuda.

        Args:
            engine_file_path: Path to the TensorRT engine file.
            device_int: CUDA device index.
        """
        if not PYCUDA_TRT_AVAILABLE:
            raise RuntimeError(
                "pycuda and tensorrt are required. "
                "Install with: pip install pycuda tensorrt"
            )

        engine_file_path = str(engine_file_path)
        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine file not found: {engine_file_path}")

        self.device_int = device_int
        self._torch_device = torch.device(f"cuda:{device_int}")

        # Create CUDA context for this device
        self.ctx = cuda.Device(device_int).make_context()

        try:
            # Load TensorRT engine
            self.engine = _get_engine(engine_file_path)

            # Get input/output tensor names
            self.input_names = []
            self.output_names = []

            for binding in self.engine:
                mode = self.engine.get_tensor_mode(binding)
                if mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(binding)
                elif mode == trt.TensorIOMode.OUTPUT:
                    self.output_names.append(binding)

            # Helper to get safe shape (handle dynamic dimensions)
            def get_safe_shape(engine, name):
                shape = engine.get_tensor_shape(name)
                if -1 in shape:
                    # Use max profile shape for dynamic dimensions
                    profile = engine.get_tensor_profile_shape(name, 0)
                    if profile:
                        logger.debug(f"Dynamic shape for {name}: {shape} -> Max: {profile[2]}")
                        return profile[2]
                return shape

            # Get shapes and dtypes for all tensors
            self.input_shapes = {name: get_safe_shape(self.engine, name) for name in self.input_names}
            self.input_dtypes = {name: self.engine.get_tensor_dtype(name) for name in self.input_names}
            self.input_nbytes = {
                name: trt.volume(self.input_shapes[name]) * trt.nptype(self.input_dtypes[name])().itemsize
                for name in self.input_names
            }

            self.output_shapes = {name: get_safe_shape(self.engine, name) for name in self.output_names}
            self.output_dtypes = {name: self.engine.get_tensor_dtype(name) for name in self.output_names}
            self.output_nbytes = {
                name: trt.volume(self.output_shapes[name]) * trt.nptype(self.output_dtypes[name])().itemsize
                for name in self.output_names
            }

            # Allocate CUDA device memory for inputs and outputs
            self.dinputs = {name: cuda.mem_alloc(self.input_nbytes[name]) for name in self.input_names}
            self.doutputs = {name: cuda.mem_alloc(self.output_nbytes[name]) for name in self.output_names}

            # Create execution context
            self.context = self.engine.create_execution_context()

            # Create CUDA stream
            self.stream = cuda.Stream()

            # Bind tensor addresses to context
            for name in self.input_names:
                self.context.set_tensor_address(name, int(self.dinputs[name]))
            for name in self.output_names:
                self.context.set_tensor_address(name, int(self.doutputs[name]))

            # Allocate page-locked host memory for outputs (for CPU transfer)
            self.houtputs = {
                name: cuda.pagelocked_empty(
                    trt.volume(self.output_shapes[name]),
                    dtype=trt.nptype(self.output_dtypes[name])
                )
                for name in self.output_names
            }

            logger.info(f"TensorRT engine loaded successfully ({len(self.input_names)} inputs, {len(self.output_names)} outputs)")

        except Exception as e:
            self.ctx.pop()
            raise RuntimeError(f"Failed to initialize TensorRT engine: {e}")

        self.ctx.pop()

    def __call__(
        self,
        output_names: list[str] | None = None,
        return_torch: bool = True,
        **inputs: torch.Tensor | np.ndarray,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Run inference on the TensorRT engine.

        Args:
            output_names: List of output names to return. If None or empty, returns all.
            return_torch: If True, return PyTorch tensors on CUDA.
            **inputs: Named inputs as PyTorch tensors or numpy arrays.

        Returns:
            Dictionary mapping output names to tensors/arrays.
        """
        if output_names is None or len(output_names) == 0:
            output_names = self.output_names

        self.ctx.push()
        result = {}

        try:
            # Copy inputs to device memory
            for name, hinput in inputs.items():
                if name not in self.input_names:
                    continue

                if isinstance(hinput, torch.Tensor) and hinput.is_cuda and hinput.device.index == self.device_int:
                    # GPU tensor on same device - device-to-device copy (fast!)
                    hinput_con = hinput.contiguous()
                    cuda.memcpy_dtod_async(
                        self.dinputs[name],
                        hinput_con.data_ptr(),
                        self.input_nbytes[name],
                        self.stream
                    )
                else:
                    # CPU tensor or numpy array - host-to-device copy
                    if isinstance(hinput, torch.Tensor):
                        hinput = hinput.detach().cpu().numpy()
                    hinput_con = np.ascontiguousarray(hinput)
                    cuda.memcpy_htod_async(self.dinputs[name], hinput_con, self.stream)

            # Set input shapes for any inputs not provided (use prefilled)
            for name in self.input_names:
                if name not in inputs:
                    self.context.set_input_shape(name, self.input_shapes[name])

            # Execute TensorRT inference asynchronously
            self.context.execute_async_v3(self.stream.handle)

            # Copy outputs
            if return_torch:
                # Device-to-device copy to new torch tensor
                for name in output_names:
                    t = torch.zeros(
                        trt.volume(self.output_shapes[name]),
                        device=self._torch_device,
                        dtype=_numpy_to_torch_dtype(trt.nptype(self.output_dtypes[name]))
                    )
                    cuda.memcpy_dtod_async(
                        t.data_ptr(),
                        self.doutputs[name],
                        self.output_nbytes[name],
                        self.stream
                    )
                    t = t.reshape(tuple(self.output_shapes[name]))
                    result[name] = t
            else:
                # Device-to-host copy to numpy
                for name in output_names:
                    cuda.memcpy_dtoh_async(self.houtputs[name], self.doutputs[name], self.stream)
                    result[name] = self.houtputs[name].reshape(self.output_shapes[name])

            # Synchronize stream
            self.stream.synchronize()

        except Exception as e:
            logger.error(f"TensorRT execution failed: {e}")
            traceback.print_exc()
            self.ctx.pop()
            raise

        self.ctx.pop()
        return result

    def prefill(self, **inputs: torch.Tensor | np.ndarray) -> bool:
        """Pre-fill inputs/outputs with constant values.

        This copies data to the pre-allocated CUDA buffers.

        Args:
            **inputs: Named inputs as PyTorch tensors or numpy arrays.

        Returns:
            True if successful.
        """
        self.ctx.push()

        try:
            for name, hinput in inputs.items():
                # Check if it's an input or output buffer
                if name in self.input_names:
                    dst_ptr = self.dinputs[name]
                elif name in self.output_names:
                    dst_ptr = self.doutputs[name]
                else:
                    logger.warning(f"Unknown tensor name for prefill: {name}")
                    continue

                # Calculate actual bytes to copy
                if isinstance(hinput, torch.Tensor):
                    real_nbytes = hinput.numel() * hinput.element_size()
                else:
                    real_nbytes = hinput.nbytes

                # Copy to device
                if isinstance(hinput, torch.Tensor) and hinput.is_cuda and hinput.device.index == self.device_int:
                    # GPU tensor - device-to-device copy
                    hinput_con = hinput.contiguous()
                    cuda.memcpy_dtod_async(dst_ptr, hinput_con.data_ptr(), real_nbytes, self.stream)
                else:
                    # CPU tensor/numpy - host-to-device copy
                    if isinstance(hinput, torch.Tensor):
                        hinput = hinput.detach().cpu().numpy()
                    hinput_con = np.ascontiguousarray(hinput)
                    cuda.memcpy_htod_async(dst_ptr, hinput_con, self.stream)

            self.stream.synchronize()

        except Exception as e:
            logger.error(f"Prefill failed: {e}")
            traceback.print_exc()
            self.ctx.pop()
            return False

        self.ctx.pop()
        return True

    def bind(self, output_to_input: dict[str, str]) -> bool:
        """Bind output buffers directly to input buffers for zero-copy recurrence.

        This sets the input tensor address to point to the output buffer,
        eliminating any memory copies for recurrent state.

        Args:
            output_to_input: Mapping from output names to input names.

        Returns:
            True if successful.
        """
        self.ctx.push()

        try:
            for output_name, input_name in output_to_input.items():
                if output_name not in self.output_names:
                    logger.warning(f"Unknown output for bind: {output_name}")
                    continue
                if input_name not in self.input_names:
                    logger.warning(f"Unknown input for bind: {input_name}")
                    continue

                # Point input address to output buffer
                self.context.set_tensor_address(input_name, int(self.doutputs[output_name]))

        except Exception as e:
            logger.error(f"Bind failed: {e}")
            traceback.print_exc()
            self.ctx.pop()
            return False

        self.ctx.pop()
        return True

    def clear_prefill(self):
        """Clear is not needed for EngineModel - buffers persist."""
        pass

    def get_input_names(self) -> list[str]:
        """Get list of input tensor names."""
        return list(self.input_names)

    def get_output_names(self) -> list[str]:
        """Get list of output tensor names."""
        return list(self.output_names)

    def get_input_shapes(self) -> dict[str, tuple]:
        """Get input tensor shapes."""
        return {name: tuple(shape) for name, shape in self.input_shapes.items()}

    def get_output_shapes(self) -> dict[str, tuple]:
        """Get output tensor shapes."""
        return {name: tuple(shape) for name, shape in self.output_shapes.items()}

    def __del__(self):
        """Clean up CUDA resources."""
        try:
            if hasattr(self, 'ctx'):
                self.ctx.push()
                # Free CUDA memory
                for ptr in getattr(self, 'dinputs', {}).values():
                    ptr.free()
                for ptr in getattr(self, 'doutputs', {}).values():
                    ptr.free()
                self.ctx.pop()
        except Exception:
            pass

    def __repr__(self) -> str:
        """String representation."""
        r = "EngineModel(\n  Inputs=[\n"
        for name in self.input_names:
            dtype = trt.nptype(self.input_dtypes[name]).__name__
            r += f"    {name}: {dtype}{tuple(self.input_shapes[name])},\n"
        r += "  ],\n  Outputs=[\n"
        for name in self.output_names:
            dtype = trt.nptype(self.output_dtypes[name]).__name__
            r += f"    {name}: {dtype}{tuple(self.output_shapes[name])},\n"
        r += "  ]\n)"
        return r


