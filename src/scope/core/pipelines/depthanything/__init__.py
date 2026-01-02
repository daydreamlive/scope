"""DepthAnything pipeline for video depth estimation.

This module provides a pipeline for temporally consistent depth estimation
on video frames using Video-Depth-Anything.
"""

from .model import VideoDepthAnythingModel
from .pipeline import DepthAnythingPipeline

__all__ = ["DepthAnythingPipeline", "VideoDepthAnythingModel"]
