"""
Shared VACE (Visual Adaptive Conditioning Enhancement) components.

Provides VACE model, utilities, and blocks for reference image conditioning
and structural guidance (depth, flow, pose, etc.) across all Wan2.1 pipelines.
"""

from .blocks.vace_encoding import VaceEncodingBlock
from .models.attention_blocks import (
    create_base_attention_block_class,
    create_vace_attention_block_class,
)
from .models.causal_vace_model import CausalVaceWanModel
from .utils.encoding import (
    decode_vace_latent,
    extract_depth_chunk,
    load_and_prepare_reference_images,
    preprocess_depth_frames,
    vace_encode_frames,
    vace_encode_masks,
    vace_latent,
)
from .utils.weight_loader import load_vace_weights_only

__all__ = [
    "CausalVaceWanModel",
    "create_vace_attention_block_class",
    "create_base_attention_block_class",
    "vace_encode_frames",
    "vace_encode_masks",
    "vace_latent",
    "load_and_prepare_reference_images",
    "decode_vace_latent",
    "preprocess_depth_frames",
    "extract_depth_chunk",
    "load_vace_weights_only",
    "VaceEncodingBlock",
]
