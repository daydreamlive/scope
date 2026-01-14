"""Scope Pose TensorRT Plugin.

TensorRT-accelerated pose estimation preprocessor for VACE/ControlNet conditioning.

Example:
    ```python
    from scope_pose_tensorrt import PoseTensorRTPreprocessor

    # Initialize preprocessor (lazy loads on first use)
    preprocessor = PoseTensorRTPreprocessor()

    # Process video frames
    # Input: [B, C, F, H, W] or [C, F, H, W] in [0, 1] range
    pose_maps = preprocessor(input_frames)
    # Output: [1, C, F, H, W] in [-1, 1] range

    # Use with VACE-enabled pipeline
    pipeline(vace_input_frames=pose_maps, ...)
    ```
"""

from .preprocessor import PoseTensorRTPreprocessor

__version__ = "0.1.0"
__all__ = ["PoseTensorRTPreprocessor"]


# Register with Scope plugin system if available
try:
    from scope.core.plugins import hookimpl

    @hookimpl
    def register_preprocessors(register):
        """Register pose preprocessor with Scope."""
        register("pose", "Pose", PoseTensorRTPreprocessor)

except ImportError:
    # Scope not installed - plugin can still be used standalone
    pass
