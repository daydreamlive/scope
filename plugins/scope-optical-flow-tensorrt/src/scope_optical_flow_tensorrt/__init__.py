"""Scope Optical Flow TensorRT Plugin.

TensorRT-accelerated optical flow preprocessor for VACE/ControlNet conditioning.
Uses RAFT (Recurrent All-Pairs Field Transforms) model for flow estimation.

Example:
    ```python
    from scope_optical_flow_tensorrt import OpticalFlowTensorRTPreprocessor

    # Initialize preprocessor (lazy loads on first use)
    preprocessor = OpticalFlowTensorRTPreprocessor()

    # Process video frames
    # Input: [B, C, F, H, W] or [C, F, H, W] in [0, 1] range
    flow_maps = preprocessor(input_frames)
    # Output: [1, C, F, H, W] in [-1, 1] range
    ```
"""

from .preprocessor import OpticalFlowTensorRTPreprocessor

__version__ = "0.1.0"
__all__ = ["OpticalFlowTensorRTPreprocessor"]


# Register with Scope plugin system if available
try:
    from scope.core.plugins import hookimpl

    @hookimpl
    def register_preprocessors(register):
        """Register optical flow preprocessor with Scope."""
        register("flow", "Flow", OpticalFlowTensorRTPreprocessor)

except ImportError:
    # Scope not installed - plugin can still be used standalone
    pass
