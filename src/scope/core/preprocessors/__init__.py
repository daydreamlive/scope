"""Preprocessors for video processing pipelines.

This module contains external model wrappers that can be used to preprocess
video input before it enters the main diffusion pipeline.
"""

from .async_depth_preprocessor import DepthPreprocessorClient, DepthResult

# Re-export VideoDepthAnything from its new location in the depthanything pipeline
# for backwards compatibility
from scope.core.pipelines.depthanything import VideoDepthAnythingModel as VideoDepthAnything

__all__ = ["VideoDepthAnything", "DepthPreprocessorClient", "DepthResult"]
