"""Shared utilities for LoRA strategy implementations.

This module provides common utility functions used across multiple LoRA strategies
to avoid code duplication.
"""

import logging
import os
import re
from typing import Any

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


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

        # Protect layer name patterns by temporarily replacing them
        # These should keep their underscores: cross_attn, self_attn
        # Use placeholders without underscores to avoid them being converted to dots
        protected_patterns = {
            "cross_attn": "<<CROSSATTN>>",
            "self_attn": "<<SELFATTN>>",
        }

        for pattern, placeholder in protected_patterns.items():
            key = key.replace(pattern, placeholder)

        # Convert underscores to dots for block/layer numbering
        key = re.sub(r"_(\d+)_", r".\1.", key)

        # Convert remaining underscores to dots
        key = key.replace("_", ".")

        # Restore protected patterns
        for pattern, placeholder in protected_patterns.items():
            key = key.replace(placeholder, pattern)

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


def standardize_lora_for_peft(
    lora_path: str,
) -> dict[str, torch.Tensor] | None:
    """
    Standardize LoRA formats to PEFT-compatible format.

    Handles LoRAs with:
    - lora_up/lora_down naming (instead of lora_A/lora_B)
    - lora_unet_* prefix (instead of diffusion_model.*)
    - Separate alpha tensors

    Args:
        lora_path: Path to original LoRA file

    Returns:
        Converted state dict, or None if no conversion needed
    """
    lora_state = load_lora_weights(lora_path)

    needs_conversion = False
    has_lora_up_down = any(
        ".lora_up.weight" in k or ".lora_down.weight" in k for k in lora_state.keys()
    )
    has_lora_unet_prefix = any(k.startswith("lora_unet_") for k in lora_state.keys())
    has_peft_format = any(
        ".lora_A.weight" in k or ".lora_B.weight" in k for k in lora_state.keys()
    )
    has_diffusion_model_prefix = any(
        k.startswith("diffusion_model.") for k in lora_state.keys()
    )

    if has_lora_up_down or has_lora_unet_prefix:
        needs_conversion = True

    if not needs_conversion:
        if not has_diffusion_model_prefix and has_peft_format:
            needs_conversion = True

    if not needs_conversion:
        return None

    converted_state = {}
    processed_keys = set()

    for lora_key in lora_state.keys():
        if lora_key in processed_keys:
            continue

        pair_result = find_lora_pair(lora_key, lora_state)
        if pair_result is None:
            continue

        base_key, alpha_key, lora_A, lora_B = pair_result

        if ".lora_up.weight" in lora_key:
            processed_keys.add(lora_key)
            processed_keys.add(f"{base_key}.lora_down.weight")
        elif ".lora_B.weight" in lora_key:
            processed_keys.add(lora_key)
            processed_keys.add(f"{base_key}.lora_A.weight")

        # Extract alpha and compute pre-scaling
        # PEFT doesn't use alpha from state_dict, so we must pre-scale the weights
        alpha = None
        if alpha_key and alpha_key in lora_state:
            alpha_tensor = lora_state[alpha_key]
            alpha = alpha_tensor.item() if alpha_tensor.numel() == 1 else None

        rank = lora_A.shape[0]
        scale_factor = (alpha / rank) if alpha is not None else 1.0

        # Apply alpha/rank scaling to lora_B (up weight) to match standard LoRA behavior
        # This ensures PEFT merges the weights with correct magnitude
        lora_B_scaled = lora_B * scale_factor

        normalized_key = normalize_lora_key(base_key)

        if not normalized_key.startswith("diffusion_model."):
            normalized_key = f"diffusion_model.{normalized_key}"

        # Store pre-scaled weights (no alpha tensor needed)
        converted_state[f"{normalized_key}.lora_A.weight"] = lora_A
        converted_state[f"{normalized_key}.lora_B.weight"] = lora_B_scaled

    logger.debug(
        f"standardize_lora_for_peft: Converted {len(converted_state)} keys from {lora_path}"
    )

    return converted_state


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
    import logging

    logger = logging.getLogger(__name__)

    key_map = {}
    # Check for PEFT-wrapped model: either has base_model prefix OR has .base_layer suffix
    # After first LoRA, keys are like "blocks.0.self_attn.q.base_layer.weight" (no base_model prefix)
    is_peft_wrapped = any(
        k.startswith("base_model.") or ".base_layer.weight" in k
        for k in model_state_dict.keys()
    )

    logger.debug(f"build_key_map: Model has {len(model_state_dict)} keys")
    logger.debug(f"build_key_map: is_peft_wrapped={is_peft_wrapped}")

    # Sample some keys for debugging
    sample_keys = list(model_state_dict.keys())[:5]
    logger.debug(f"build_key_map: Sample model keys: {sample_keys}")

    weight_keys_count = 0
    peft_base_layer_count = 0
    for k in model_state_dict.keys():
        if k.endswith(".weight"):
            weight_keys_count += 1
            base_key = k[: -len(".weight")]
            key_map[base_key] = k

            if is_peft_wrapped and k.endswith(".base_layer.weight"):
                peft_base_layer_count += 1
                # Handle two PEFT key formats:
                # 1. "base_model.model.blocks.0.self_attn.q.base_layer.weight" (with base_model prefix)
                # 2. "blocks.0.self_attn.q.base_layer.weight" (without base_model prefix, after first LoRA)
                if k.startswith("base_model.model."):
                    # Format 1: Strip base_model.model prefix and .base_layer suffix
                    peft_stripped = k[
                        len("base_model.model.") : -len(".base_layer.weight")
                    ]
                else:
                    # Format 2: Just strip .base_layer suffix
                    peft_stripped = k[: -len(".base_layer.weight")]

                key_map[peft_stripped] = k
                key_map[f"diffusion_model.{peft_stripped}"] = k
            else:
                key_map[f"diffusion_model.{base_key}"] = k

    logger.debug(f"build_key_map: Found {weight_keys_count} .weight keys")
    logger.debug(
        f"build_key_map: Found {peft_base_layer_count} PEFT base_layer.weight keys"
    )
    logger.debug(f"build_key_map: Built key_map with {len(key_map)} entries")

    # Sample some key_map entries for debugging
    sample_map_entries = list(key_map.items())[:5]
    logger.debug(f"build_key_map: Sample key_map entries: {sample_map_entries}")

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
    import logging

    logger = logging.getLogger(__name__)

    lora_mapping = {}
    processed_keys = set()
    unmatched_keys = []
    matched_keys = []

    # Build model key map using the shared utility that handles PEFT-wrapped models
    model_key_map = build_key_map(model_state)

    logger.debug(f"parse_lora_weights: LoRA state has {len(lora_state)} keys")
    sample_lora_keys = list(lora_state.keys())[:5]
    logger.debug(f"parse_lora_weights: Sample LoRA keys: {sample_lora_keys}")

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
        if len(matched_keys) < 3:  # Only log first few to avoid spam
            logger.debug(
                f"parse_lora_weights: LoRA base_key='{base_key}' -> normalized_key='{normalized_key}'"
            )

        # Find matching model key
        model_key = model_key_map.get(normalized_key)
        if model_key is None:
            model_key = model_key_map.get(f"diffusion_model.{normalized_key}")

        if model_key is None:
            unmatched_keys.append((base_key, normalized_key))
            if len(unmatched_keys) <= 3:  # Only log first few to avoid spam
                logger.debug(
                    f"parse_lora_weights: No match for base_key='{base_key}', normalized_key='{normalized_key}'"
                )
                # Check what similar keys exist in model_key_map
                similar_keys = [
                    k
                    for k in model_key_map.keys()
                    if normalized_key.split(".")[-1] in k
                    or (
                        len(normalized_key.split(".")) > 0
                        and normalized_key.split(".")[-2] in k
                    )
                ][:3]
                if similar_keys:
                    logger.debug(
                        f"parse_lora_weights: Similar keys in model_key_map: {similar_keys}"
                    )
            continue

        matched_keys.append((base_key, normalized_key, model_key))
        if len(matched_keys) <= 3:  # Only log first few to avoid spam
            logger.debug(
                f"parse_lora_weights: Matched base_key='{base_key}' -> model_key='{model_key}'"
            )

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

    logger.debug(
        f"parse_lora_weights: Matched {len(matched_keys)} keys, unmatched {len(unmatched_keys)} keys"
    )
    if unmatched_keys:
        logger.debug(
            f"parse_lora_weights: First 5 unmatched keys: {unmatched_keys[:5]}"
        )

    return lora_mapping
