"""
Shared VACE (Visual Adaptive Conditioning Enhancement) components.

Provides VACE model, utilities, and blocks for reference image conditioning
and structural guidance (depth, flow, pose, etc.) across all Wan2.1 pipelines.
"""

from .mixin import VACEEnabledPipeline
from .models.causal_vace_model import CausalVaceWanModel
from .utils.weight_loader import load_vace_weights_only

__all__ = [
    "CausalVaceWanModel",
    "load_vace_weights_only",
    "VACEEnabledPipeline",
]
