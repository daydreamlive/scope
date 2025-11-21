"""
Permanent merge LoRA strategy for WAN models.

Merges LoRA weights into model weights at load time using PEFT's merge_and_unload(),
providing zero inference overhead. LoRA scales are fixed at load time and cannot be
updated at runtime. Ideal for production deployment where maximum FPS is critical.
"""

import os
import tempfile
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from ..utils import find_lora_pair, normalize_lora_key
from .peft_lora import PeftLoRAStrategy

__all__ = ["PermanentMergeLoRAStrategy"]


def convert_community_lora_to_peft_format(
    lora_path: str, model: torch.nn.Module
) -> str:
    """
    Convert community LoRA formats to PEFT-compatible format.

    Handles LoRAs with:
    - lora_up/lora_down naming (instead of lora_A/lora_B)
    - lora_unet_* prefix (instead of diffusion_model.*)
    - Separate alpha tensors

    Args:
        lora_path: Path to original LoRA file
        model: Model to check for key compatibility

    Returns:
        Path to converted LoRA file (temp file) or original path if no conversion needed
    """
    lora_state = (
        load_file(lora_path)
        if lora_path.endswith(".safetensors")
        else torch.load(lora_path, map_location="cpu")
    )

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
        return lora_path

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

    temp_fd, temp_path = tempfile.mkstemp(
        suffix=".safetensors", prefix="lora_converted_"
    )
    os.close(temp_fd)

    try:
        save_file(converted_state, temp_path)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise RuntimeError(
            f"convert_community_lora_to_peft_format: Failed to save converted LoRA: {e}"
        ) from e


class PermanentMergeLoRAStrategy:
    """
    Manages LoRA adapters via permanent weight merging at load time.

    Uses PEFT's merge_and_unload() to merge LoRA weights directly into model
    parameters, eliminating inference overhead. LoRA scales are fixed at load time
    and cannot be updated at runtime.

    Ideal for production deployment where maximum FPS is critical.
    Compatible with FP8 quantization.
    """

    @staticmethod
    def load_adapter(
        model: torch.nn.Module,
        lora_path: str,
        strength: float = 1.0,
    ) -> str:
        """
        Load and permanently merge LoRA adapter into model weights.

        The adapter is loaded with PEFT, merged into the base model weights,
        then the PEFT wrapper is removed. The resulting model has the LoRA
        effect permanently baked in with zero inference overhead.

        NOTE: For loading multiple LoRAs, use load_adapters_from_list() instead.
        Calling this method multiple times will cause key mismatches.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Strength multiplier for LoRA effect

        Returns:
            The lora_path (used as identifier)

        Raises:
            FileNotFoundError: If the LoRA file does not exist
        """
        converted_lora_path = None
        try:
            converted_lora_path = convert_community_lora_to_peft_format(
                lora_path=lora_path, model=model
            )

            adapter_name = PeftLoRAStrategy.load_adapter(
                model=model, lora_path=converted_lora_path, strength=strength
            )
        finally:
            if converted_lora_path and converted_lora_path != lora_path:
                if os.path.exists(converted_lora_path):
                    os.unlink(converted_lora_path)

        peft_model = PeftLoRAStrategy._get_peft_model(model)
        if peft_model is None:
            return str(lora_path)

        target_model = (
            peft_model._orig_mod if hasattr(peft_model, "_orig_mod") else peft_model
        )

        all_adapter_names = [adapter_name]
        if hasattr(target_model, "peft_config"):
            existing_adapters = list(target_model.peft_config.keys())
            for existing_adapter in existing_adapters:
                if existing_adapter not in all_adapter_names:
                    all_adapter_names.append(existing_adapter)

        merged_model = target_model.merge_and_unload(
            safe_merge=True, adapter_names=all_adapter_names
        )

        model.__class__ = merged_model.__class__
        model.__dict__ = merged_model.__dict__

        if hasattr(model, "peft_config"):
            delattr(model, "peft_config")
        if hasattr(model, "active_adapter"):
            delattr(model, "active_adapter")
        if hasattr(model, "peft_type"):
            delattr(model, "peft_type")

        if model in PeftLoRAStrategy._peft_models:
            del PeftLoRAStrategy._peft_models[model]

        return str(lora_path)

    @staticmethod
    def load_adapters_from_list(
        model: torch.nn.Module,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters by permanently merging into model weights.

        This implementation loads all LoRAs into PEFT first, then merges them all
        at once to properly support multiple LoRAs without key conflicts.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: path, scale
        """
        loaded_adapters = []

        if not lora_configs:
            return loaded_adapters

        adapter_names = []
        converted_files = []

        try:
            for lora_config in lora_configs:
                lora_path = lora_config.get("path")
                if not lora_path:
                    continue

                scale = lora_config.get("scale", 1.0)

                try:
                    converted_lora_path = convert_community_lora_to_peft_format(
                        lora_path=lora_path, model=model
                    )

                    if converted_lora_path != lora_path:
                        converted_files.append(converted_lora_path)

                    adapter_name = PeftLoRAStrategy.load_adapter(
                        model=model, lora_path=converted_lora_path, strength=scale
                    )
                    adapter_names.append(adapter_name)
                    loaded_adapters.append({"path": str(lora_path), "scale": scale})

                except FileNotFoundError as e:
                    raise RuntimeError(
                        f"{logger_prefix}LoRA loading failed. File not found: {lora_path}. "
                        f"Ensure the file exists in the models/lora/ directory."
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"{logger_prefix}LoRA loading failed. Pipeline cannot start without all configured LoRAs. "
                        f"Error: {e}"
                    ) from e
        finally:
            for temp_file in converted_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

        if adapter_names:
            peft_model = PeftLoRAStrategy._get_peft_model(model)
            if peft_model is None:
                return loaded_adapters

            target_model = (
                peft_model._orig_mod if hasattr(peft_model, "_orig_mod") else peft_model
            )

            all_adapter_names = adapter_names.copy()
            if hasattr(target_model, "peft_config"):
                existing_adapters = list(target_model.peft_config.keys())
                for existing_adapter in existing_adapters:
                    if existing_adapter not in all_adapter_names:
                        all_adapter_names.append(existing_adapter)

            merged_model = target_model.merge_and_unload(
                safe_merge=True, adapter_names=all_adapter_names
            )

            model.__class__ = merged_model.__class__
            model.__dict__ = merged_model.__dict__

            if hasattr(model, "peft_config"):
                delattr(model, "peft_config")
            if hasattr(model, "active_adapter"):
                delattr(model, "active_adapter")
            if hasattr(model, "peft_type"):
                delattr(model, "peft_type")

            if model in PeftLoRAStrategy._peft_models:
                del PeftLoRAStrategy._peft_models[model]

        return loaded_adapters

    @staticmethod
    def update_adapter_scales(
        model: torch.nn.Module,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: list[dict[str, Any]],
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """
        Update scales for loaded LoRA adapters at runtime.

        WARNING: This operation is NOT SUPPORTED for permanent merge mode.
        LoRA weights are permanently baked into the model at load time and cannot
        be updated without reloading the model.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages

        Returns:
            Unchanged loaded_adapters list (updates not supported)
        """
        return loaded_adapters
