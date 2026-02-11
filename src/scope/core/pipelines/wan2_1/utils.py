import os

import torch
from safetensors.torch import load_file as load_safetensors

from scope.core.pipelines.longlive.modules.kv_cache_manager import KVCacheManager


def initialize_kv_cache_manager(
    generator,
    batch_size,
    dtype,
    device,
    local_attn_size,
    frame_seq_length,
    sink_size=0,
    existing_manager=None,
):
    """
    Create or reset a KVCacheManager with ring buffer cache.

    Args:
        generator: The WanDiffusionWrapper containing the model.
        batch_size: Batch size (typically 1).
        dtype: Data type for cache tensors.
        device: Device for cache tensors.
        local_attn_size: Number of frames in the local attention window.
        frame_seq_length: Number of tokens per frame.
        sink_size: Number of frames to use as sink tokens.
        existing_manager: Existing KVCacheManager to reset (avoids reallocation).

    Returns:
        KVCacheManager instance with cache buffers bound to the model.
    """
    num_layers = len(generator.model.blocks)
    num_heads = generator.model.num_heads
    head_dim = generator.model.dim // num_heads

    if local_attn_size != -1:
        cache_size = local_attn_size * frame_seq_length
    else:
        cache_size = 32760

    sink_tokens = sink_size * frame_seq_length

    # Reuse existing manager if shapes match
    if (
        existing_manager is not None
        and existing_manager.cache_size == cache_size
        and existing_manager.num_layers == num_layers
    ):
        existing_manager.full_reset()
        generator.model.set_cache(existing_manager.cache_k, existing_manager.cache_v)
        return existing_manager

    # Create new manager
    manager = KVCacheManager(
        num_layers=num_layers,
        batch_size=batch_size,
        cache_size=cache_size,
        num_heads=num_heads,
        head_dim=head_dim,
        sink_tokens=sink_tokens,
        frame_seqlen=frame_seq_length,
        device=device,
        dtype=dtype,
    )

    # Bind cache buffers to the model
    generator.model.set_cache(manager.cache_k, manager.cache_v)

    return manager


def initialize_kv_cache(
    generator,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    local_attn_size: int,
    frame_seq_length: int,
    kv_cache_existing: list[dict] | None = None,
    reset_indices: bool = True,
    zero_cache: bool = True,
):
    kv_cache = []

    # Calculate KV cache size
    if local_attn_size != -1:
        # Use the local attention size to compute the KV cache size
        kv_cache_size = local_attn_size * frame_seq_length
    else:
        # Use the default KV cache size
        kv_cache_size = 32760

    # Get transformer config
    num_transformer_blocks = len(generator.model.blocks)
    num_heads = generator.model.num_heads
    dim = generator.model.dim
    k_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]
    v_shape = [batch_size, kv_cache_size, num_heads, dim // num_heads]

    # Check if we can reuse existing cache
    if (
        kv_cache_existing
        and len(kv_cache_existing) > 0
        and list(kv_cache_existing[0]["k"].shape) == k_shape
        and list(kv_cache_existing[0]["v"].shape) == v_shape
    ):
        for i in range(num_transformer_blocks):
            if zero_cache:
                kv_cache_existing[i]["k"].zero_()
                kv_cache_existing[i]["v"].zero_()

            if reset_indices:
                kv_cache_existing[i]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
                kv_cache_existing[i]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )

        return kv_cache_existing
    else:
        # Create new cache
        for _ in range(num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
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
