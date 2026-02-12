import os

import torch
from safetensors.torch import load_file as load_safetensors


def initialize_kv_cache(
    generator,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    local_attn_size: int,
    frame_seq_length: int,
    num_frame_per_block: int = 3,
    kv_cache_existing: list[dict] | None = None,
):
    kv_cache = []

    # Cache stores context only: (local_attn_size - num_frame_per_block) frames.
    # During attention, the new num_frame_per_block frames are concatenated to reach
    # the full local_attn_size window.
    if local_attn_size != -1:
        kv_cache_size = (local_attn_size - num_frame_per_block) * frame_seq_length
    else:
        kv_cache_size = 32760

    # Get transformer config
    num_transformer_blocks = len(generator.model.blocks)
    num_heads = generator.model.num_heads
    dim = generator.model.dim
    k_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]
    v_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]

    # Check if we can reuse existing cache tensors (same shape)
    if (
        kv_cache_existing
        and len(kv_cache_existing) > 0
        and list(kv_cache_existing[0]["k"].shape) == k_shape
        and list(kv_cache_existing[0]["v"].shape) == v_shape
    ):
        for i in range(num_transformer_blocks):
            kv_cache_existing[i]["k"].zero_()
            kv_cache_existing[i]["v"].zero_()
        return kv_cache_existing
    else:
        # Create new cache -- just k and v tensors, no index tracking
        for _ in range(num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                }
            )
        return kv_cache


def initialize_crossattn_cache(
    generator,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    crossattn_cache_existing: list[dict] | None = None,
):
    crossattn_cache = []

    # Get transformer config
    num_transformer_blocks = len(generator.model.blocks)
    num_heads = generator.model.num_heads
    dim = generator.model.dim
    k_shape = [batch_size, 512, num_heads, dim // num_heads]
    v_shape = [batch_size, 512, num_heads, dim // num_heads]

    # Check if we can reuse existing cache
    if (
        crossattn_cache_existing
        and len(crossattn_cache_existing) > 0
        and list(crossattn_cache_existing[0]["k"].shape) == k_shape
        and list(crossattn_cache_existing[0]["v"].shape) == v_shape
    ):
        for i in range(num_transformer_blocks):
            crossattn_cache_existing[i]["k"].zero_()
            crossattn_cache_existing[i]["v"].zero_()
            crossattn_cache_existing[i]["is_init"] = False
        return crossattn_cache_existing
    else:
        # Create new cache
        for _ in range(num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                    "is_init": False,
                }
            )
        return crossattn_cache


def load_state_dict(weights_path: str) -> dict:
    """Load weights with automatic format detection."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    if weights_path.endswith(".safetensors"):
        # Load from safetensors and convert keys
        state_dict = load_safetensors(weights_path)

    elif weights_path.endswith(".pth") or weights_path.endswith(".pt"):
        # Load from PyTorch format (assume already in correct format)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    else:
        raise ValueError(
            f"Unsupported file format. Expected .safetensors, .pth, or .pt, got: {weights_path}"
        )

    return state_dict
