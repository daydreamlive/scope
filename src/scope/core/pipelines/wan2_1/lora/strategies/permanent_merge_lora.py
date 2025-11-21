"""
Permanent merge LoRA strategy for WAN models.

Merges LoRA weights into model weights at load time using PEFT's merge_and_unload(),
providing zero inference overhead. LoRA scales are fixed at load time and cannot be
updated at runtime. Ideal for production deployment where maximum FPS is critical.
"""

from typing import Any

import torch
from safetensors.torch import load_file

from ..utils import find_lora_pair, normalize_lora_key
from .peft_lora import PeftLoRAStrategy

__all__ = ["PermanentMergeLoRAStrategy"]


def standardize_lora_for_peft(
    lora_path: str, model: torch.nn.Module
) -> dict[str, torch.Tensor] | None:
    """
    Standardize LoRA formats to PEFT-compatible format.

    Handles LoRAs with:
    - lora_up/lora_down naming (instead of lora_A/lora_B)
    - lora_unet_* prefix (instead of diffusion_model.*)
    - Separate alpha tensors

    Args:
        lora_path: Path to original LoRA file
        model: Model to check for key compatibility

    Returns:
        Converted state dict, or None if no conversion needed
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

    return converted_state


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
    def _load_adapter_from_state_dict_or_path(
        model: torch.nn.Module,
        lora_path: str,
        converted_state: dict[str, torch.Tensor] | None,
        strength: float = 1.0,
    ) -> str:
        """
        Load adapter either from converted state dict or original path.

        Args:
            model: PyTorch model
            lora_path: Path to original LoRA file
            converted_state: Pre-converted state dict, or None to load from path
            strength: Strength multiplier for LoRA effect

        Returns:
            The adapter name
        """
        if converted_state is None:
            # No conversion needed, load directly from path
            return PeftLoRAStrategy.load_adapter(
                model=model, lora_path=lora_path, strength=strength
            )

        # Load from in-memory state dict
        from pathlib import Path

        from ..utils import parse_lora_weights, sanitize_adapter_name

        adapter_name = sanitize_adapter_name(Path(lora_path).stem)

        # Parse the converted state dict through PeftLoRAStrategy
        # to inject LoRA layers (this handles the PEFT wrapping)
        model_state = model.state_dict()
        lora_mapping = parse_lora_weights(converted_state, model_state)

        # Inject PEFT LoRA layers
        PeftLoRAStrategy._inject_lora_layers(
            model, lora_mapping, adapter_name, strength
        )

        return adapter_name

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
        # Convert to PEFT format if needed (in-memory, no disk I/O)
        converted_state = standardize_lora_for_peft(lora_path=lora_path, model=model)

        adapter_name = PermanentMergeLoRAStrategy._load_adapter_from_state_dict_or_path(
            model=model,
            lora_path=lora_path,
            converted_state=converted_state,
            strength=strength,
        )

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

        for lora_config in lora_configs:
            lora_path = lora_config.get("path")
            if not lora_path:
                continue

            scale = lora_config.get("scale", 1.0)

            try:
                # Convert to PEFT format if needed (in-memory, no disk I/O)
                converted_state = standardize_lora_for_peft(
                    lora_path=lora_path, model=model
                )

                adapter_name = (
                    PermanentMergeLoRAStrategy._load_adapter_from_state_dict_or_path(
                        model=model,
                        lora_path=lora_path,
                        converted_state=converted_state,
                        strength=scale,
                    )
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
