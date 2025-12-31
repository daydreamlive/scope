"""Preprocessors for video processing pipelines.

This module contains external model wrappers that can be used to preprocess
video input before it enters the main diffusion pipeline.
"""

from .async_depth_preprocessor import DepthPreprocessorClient, DepthResult
from .video_depth_anything import VideoDepthAnything

__all__ = ["VideoDepthAnything", "DepthPreprocessorClient", "DepthResult"]
