"""Shared utilities for LoRA strategy implementations.

This module provides common utility functions used across multiple LoRA strategies
to avoid code duplication.
"""

import os
import re
from typing import Any

import torch
from safetensors.torch import load_file


def sanitize_adapter_name(adapter_name: str) -> str:
    """
    Sanitize adapter name to be valid for PyTorch module names.

    PyTorch module names cannot contain periods (.), so we replace them
    with underscores. Also removes other potentially problematic characters.

    Args:
        adapter_name: Original adapter name (may contain periods, slashes, etc.)

    Returns:
        Sanitized adapter name safe for PyTorch module registration
    """
    # Replace periods with underscores (PyTorch doesn't allow periods in module names)
    sanitized = adapter_name.replace(".", "_")
    # Remove any other potentially problematic characters
    sanitized = (
        sanitized.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return sanitized


def normalize_lora_key(lora_base_key: str) -> str:
    """
    Normalize LoRA base key to match model state dict format.

    Handles various LoRA naming conventions:
    - lora_unet_blocks_0_cross_attn_k -> blocks.0.cross_attn.k
    - diffusion_model.blocks.0.cross_attn.k -> blocks.0.cross_attn.k
    - blocks.0.cross_attn.k -> blocks.0.cross_attn.k

    Args:
        lora_base_key: Base key from LoRA file (without .lora_A/B/up/down.weight)

    Returns:
        Normalized key that matches model state dict format
    """
    # Handle lora_unet_* format (with underscores)
    if lora_base_key.startswith("lora_unet_"):
        # Remove lora_unet_ prefix
        key = lora_base_key[len("lora_unet_") :]
        # Convert underscores to dots for block/layer numbering
        key = re.sub(r"_(\d+)_", r".\1.", key)
        # Convert remaining underscores to dots for layer names
        key = key.replace("_", ".")
        return key

    # Handle diffusion_model prefix
    if lora_base_key.startswith("diffusion_model."):
        return lora_base_key[len("diffusion_model.") :]

    return lora_base_key


def load_lora_weights(lora_path: str) -> dict[str, torch.Tensor]:
    """
    Load LoRA weights from .safetensors or .bin file.

    Args:
        lora_path: Path to LoRA file (.safetensors or .bin)

    Returns:
        Dictionary mapping parameter names to tensors

    Raises:
        FileNotFoundError: If the LoRA file does not exist
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"load_lora_weights: LoRA file not found: {lora_path}")

    if lora_path.endswith(".safetensors"):
        return load_file(lora_path)
    else:
        return torch.load(lora_path, map_location="cpu")


def find_lora_pair(
    lora_key: str, lora_state: dict[str, torch.Tensor]
) -> tuple[str, str, torch.Tensor, torch.Tensor] | None:
    """
    Find LoRA A/B or up/down weight pair from a LoRA key.

    Args:
        lora_key: LoRA key to check (e.g., "blocks.0.attn.q.lora_up.weight")
        lora_state: Full LoRA state dictionary

    Returns:
        Tuple of (base_key, alpha_key, lora_A, lora_B) if pair found, None otherwise
        lora_A is the down/input matrix, lora_B is the up/output matrix
    """
    lora_A, lora_B, alpha_key = None, None, None
    base_key = None

    if ".lora_up.weight" in lora_key:
        base_key = lora_key.replace(".lora_up.weight", "")
        lora_down_key = f"{base_key}.lora_down.weight"
        alpha_key = f"{base_key}.alpha"
        if lora_down_key in lora_state:
            lora_B = lora_state[lora_key]  # lora_up is the B/up matrix
            lora_A = lora_state[lora_down_key]  # lora_down is the A/down matrix

    elif ".lora_B.weight" in lora_key:
        base_key = lora_key.replace(".lora_B.weight", "")
        lora_A_key = f"{base_key}.lora_A.weight"
        alpha_key = f"{base_key}.alpha"
        if lora_A_key in lora_state:
            lora_B = lora_state[lora_key]
            lora_A = lora_state[lora_A_key]

    if base_key is None or lora_A is None or lora_B is None:
        return None

    return (base_key, alpha_key, lora_A, lora_B)


def calculate_lora_scale(alpha: float | None, rank: int) -> float:
    """
    Calculate LoRA scale from alpha and rank.

    Args:
        alpha: LoRA alpha value (optional)
        rank: LoRA rank (dimension of low-rank matrices)

    Returns:
        Scale value (alpha / rank if alpha provided, else 1.0)
    """
    if alpha is not None:
        return alpha / rank
    return 1.0


def build_key_map(model_state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """
    Build mapping from LoRA keys to model state dict keys.

    Handles multiple key formats:
    - Standard: LoRA keys like "blocks.0.attn.k" -> model "blocks.0.attn.k.weight"
    - ComfyUI: LoRA keys like "diffusion_model.blocks.0.attn.k" -> model "blocks.0.attn.k.weight"
    - PEFT-wrapped: LoRA keys like "diffusion_model.blocks.0.attn.k" -> model "base_model.model.blocks.0.attn.k.base_layer.weight"
    - Underscore format: LoRA keys like "lora_unet_blocks_0_attn_k" -> model "blocks.0.attn.k.weight"

    Args:
        model_state_dict: Model's state dict

    Returns:
        Dictionary mapping LoRA key patterns to actual model keys
    """
    key_map = {}
    is_peft_wrapped = any(k.startswith("base_model.") for k in model_state_dict.keys())

    for k in model_state_dict.keys():
        if k.endswith(".weight"):
            base_key = k[: -len(".weight")]
            key_map[base_key] = k

            if (
                is_peft_wrapped
                and k.startswith("base_model.model.")
                and k.endswith(".base_layer.weight")
            ):
                # Strip PEFT prefix and suffix to match LoRA keys
                peft_stripped = k[len("base_model.model.") : -len(".base_layer.weight")]
                key_map[peft_stripped] = k
                key_map[f"diffusion_model.{peft_stripped}"] = k
            else:
                key_map[f"diffusion_model.{base_key}"] = k

    return key_map


def parse_lora_weights(
    lora_state: dict[str, torch.Tensor], model_state: dict[str, torch.Tensor]
) -> dict[str, dict[str, Any]]:
    """
    Parse LoRA weights and match them to model parameters.

    Returns:
        Dict mapping model parameter names to LoRA info:
        {
            "blocks.0.self_attn.q.weight": {
                "lora_A": tensor,
                "lora_B": tensor,
                "alpha": float or None,
                "rank": int
            }
        }
    """
    lora_mapping = {}
    processed_keys = set()

    # Build model key map using the shared utility that handles PEFT-wrapped models
    model_key_map = build_key_map(model_state)

    # Iterate through LoRA keys to find A/B or up/down pairs
    for lora_key in lora_state.keys():
        if lora_key in processed_keys:
            continue

        # Find LoRA pair
        pair_result = find_lora_pair(lora_key, lora_state)
        if pair_result is None:
            continue

        base_key, alpha_key, lora_A, lora_B = pair_result

        # Mark both keys as processed
        if ".lora_up.weight" in lora_key:
            processed_keys.add(lora_key)
            processed_keys.add(f"{base_key}.lora_down.weight")
        elif ".lora_B.weight" in lora_key:
            processed_keys.add(lora_key)
            processed_keys.add(f"{base_key}.lora_A.weight")

        # Normalize the base key
        normalized_key = normalize_lora_key(base_key)

        # Find matching model key
        model_key = model_key_map.get(normalized_key)
        if model_key is None:
            model_key = model_key_map.get(f"diffusion_model.{normalized_key}")

        if model_key is None:
            continue

        # Extract alpha and rank
        alpha = None
        if alpha_key and alpha_key in lora_state:
            alpha = lora_state[alpha_key].item()

        rank = lora_A.shape[0]

        lora_mapping[model_key] = {
            "lora_A": lora_A,
            "lora_B": lora_B,
            "alpha": alpha,
            "rank": rank,
        }

    return lora_mapping
