"""Preprocessors for video processing pipelines.

This module contains external model wrappers that can be used to preprocess
video input before it enters the main diffusion pipeline.
"""

from .async_preprocessor import AsyncPreprocessorClient, PreprocessorResult

# Backward compatibility aliases
from .async_preprocessor import AsyncPreprocessorClient as DepthPreprocessorClient
from .async_preprocessor import PreprocessorResult as DepthResult

# Re-export VideoDepthAnything from its new location in the depthanything pipeline
# for backwards compatibility
from scope.core.pipelines.depthanything import VideoDepthAnythingModel as VideoDepthAnything

__all__ = ["VideoDepthAnything", "AsyncPreprocessorClient", "PreprocessorResult", "DepthPreprocessorClient", "DepthResult"]
