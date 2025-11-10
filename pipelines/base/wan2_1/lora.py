"""
LoRA utilities for WAN models using direct weight patching.
Supports local .safetensors and .bin files from models/lora/ directory.
"""
from typing import Dict, Any, List
import os
import time
import logging
from pathlib import Path
import torch
from safetensors.torch import load_file

__all__ = ["LoRAManager"]

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Manages LoRA adapters for WAN models using direct weight patching.

    This applies LoRA weights by computing the diff and merging into model weights,
    similar to ComfyUI's approach. Supports runtime scale adjustment by storing
    original weights and pre-computed diffs.
    """

    # Store state per model instance (using id(model) as key)
    _model_states = {}

    @staticmethod
    def _get_model_state(model: torch.nn.Module) -> Dict[str, Any]:
        """Get or create state dict for a model instance."""
        model_id = id(model)
        if model_id not in LoRAManager._model_states:
            LoRAManager._model_states[model_id] = {
                "original_weights": {},  # Stores original weights before any LoRA
                "loras": {},  # {path: {"diffs": {key: tensor}, "scale": float}}
                "backed_up": False  # Flag to track if originals are stored
            }
        return LoRAManager._model_states[model_id]

    @staticmethod
    def _normalize_lora_key(lora_base_key: str) -> str:
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
            key = lora_base_key[len("lora_unet_"):]
            # Convert underscores to dots for block/layer numbering
            # Pattern: blocks_N_ -> blocks.N.
            import re
            key = re.sub(r'_(\d+)_', r'.\1.', key)
            # Convert remaining underscores to dots for layer names
            key = key.replace('_', '.')
            return key

        # Handle diffusion_model prefix
        if lora_base_key.startswith("diffusion_model."):
            return lora_base_key[len("diffusion_model."):]

        return lora_base_key

    @staticmethod
    def _build_key_map(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
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
                base_key = k[:-len(".weight")]
                key_map[base_key] = k

                if is_peft_wrapped and k.startswith("base_model.model.") and k.endswith(".base_layer.weight"):
                    # Strip PEFT prefix and suffix to match LoRA keys
                    peft_stripped = k[len("base_model.model."):-len(".base_layer.weight")]
                    key_map[peft_stripped] = k
                    key_map[f"diffusion_model.{peft_stripped}"] = k
                else:
                    key_map[f"diffusion_model.{base_key}"] = k

        return key_map

    @staticmethod
    def _backup_original_weights(model: torch.nn.Module, keys_to_backup: List[str]) -> None:
        """Backup original model weights before any LoRA patching."""
        state = LoRAManager._get_model_state(model)

        if state["backed_up"]:
            return  # Already backed up

        model_state = model.state_dict()
        for key in keys_to_backup:
            if key in model_state and key not in state["original_weights"]:
                # Clone and detach to avoid issues
                state["original_weights"][key] = model_state[key].clone()

        state["backed_up"] = True
        logger.debug(f"_backup_original_weights: Backed up {len(state['original_weights'])} original weights")

    @staticmethod
    def _apply_all_loras(model: torch.nn.Module) -> None:
        """Recompute patched weights from originals + all loaded LoRA diffs."""
        state = LoRAManager._get_model_state(model)

        if not state["backed_up"]:
            return  # Nothing to apply

        model_state = model.state_dict()

        # Start from originals
        for key, original in state["original_weights"].items():
            model_state[key] = original.clone()

        # Apply all LoRA diffs with their scales
        for path, lora_info in state["loras"].items():
            scale = lora_info["scale"]
            for key, diff in lora_info["diffs"].items():
                if key in model_state:
                    model_state[key] = model_state[key] + (scale * diff).to(model_state[key].dtype)

        # Reload into model
        model.load_state_dict(model_state, strict=True)

    @staticmethod
    def load_adapter(
        model: torch.nn.Module,
        lora_path: str,
        strength: float = 1.0,
    ) -> str:
        """
        Load LoRA by computing diffs and storing them for runtime scale adjustment.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier for LoRA effect (default 1.0)

        Returns:
            The lora_path (used as identifier)

        Raises:
            FileNotFoundError: If the LoRA file does not exist

        Example:
            >>> from pipelines.base.wan2_1.lora import LoRAManager
            >>> path = LoRAManager.load_adapter(
            ...     model=pipeline.transformer,
            ...     lora_path="models/lora/my-style.safetensors",
            ...     strength=1.0
            ... )
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"load_adapter: LoRA file not found: {lora_path}")

        state = LoRAManager._get_model_state(model)

        # Load LoRA weights
        if lora_path.endswith('.safetensors'):
            lora_state = load_file(lora_path)
        else:
            lora_state = torch.load(lora_path, map_location='cpu')

        logger.debug(f"load_adapter: Loaded {len(lora_state)} keys from {lora_path}")

        model_state = model.state_dict()
        key_map = LoRAManager._build_key_map(model_state)
        logger.debug(f"load_adapter: Built key map with {len(key_map)} entries")

        # First pass: collect all affected keys and compute diffs
        lora_diffs = {}  # {model_key: diff_tensor}
        keys_to_backup = []
        applied_count = 0
        skipped_no_match = 0
        skipped_no_down = 0
        skipped_no_model_key = 0

        for lora_key in lora_state.keys():
            # Look for lora_up/lora_B weights
            if '.lora_up.weight' in lora_key:
                base_key = lora_key.replace('.lora_up.weight', '')
                down_key = f"{base_key}.lora_down.weight"
                alpha_key = f"{base_key}.alpha"
            elif '.lora_B.weight' in lora_key:
                base_key = lora_key.replace('.lora_B.weight', '')
                down_key = f"{base_key}.lora_A.weight"
                alpha_key = f"{base_key}.alpha"
            else:
                skipped_no_match += 1
                continue

            # Check if we have the matching down weight
            if down_key not in lora_state:
                skipped_no_down += 1
                if applied_count == 0:
                    logger.warning(f"load_adapter: Missing down weight for {lora_key}, expected {down_key}")
                continue

            # Normalize the base key to match model format
            normalized_key = LoRAManager._normalize_lora_key(base_key)

            # Find the model weight key (try normalized key first, then with diffusion_model prefix)
            model_key = key_map.get(normalized_key)
            if model_key is None:
                model_key = key_map.get(f"diffusion_model.{normalized_key}")

            if model_key is None or model_key not in model_state:
                skipped_no_model_key += 1
                if skipped_no_model_key <= 3:
                    logger.debug(f"load_adapter: No model key found for base_key={base_key}, normalized={normalized_key}")
                continue

            # Get alpha if present
            alpha = lora_state.get(alpha_key)
            if alpha is not None:
                alpha = alpha.item()

            # Compute the LoRA diff (without strength, we'll apply that later)
            lora_up = lora_state[lora_key]
            lora_down = lora_state[down_key]
            original_weight = model_state[model_key]

            # Move LoRA weights to same device as model weight
            lora_up = lora_up.to(device=original_weight.device)
            lora_down = lora_down.to(device=original_weight.device)

            # Compute scale: alpha / rank
            rank = lora_down.shape[0]
            if alpha is not None:
                scale = alpha / rank
            else:
                scale = 1.0

            # Compute LoRA diff: up @ down (with alpha scaling baked in)
            lora_diff = torch.mm(
                lora_up.float().flatten(start_dim=1),
                lora_down.float().flatten(start_dim=1)
            ).reshape(original_weight.shape)

            # Store the scaled diff (alpha is baked in, strength will be applied at runtime)
            lora_diffs[model_key] = (scale * lora_diff).to(original_weight.dtype)
            keys_to_backup.append(model_key)
            applied_count += 1

        # Backup original weights before first LoRA
        if keys_to_backup:
            LoRAManager._backup_original_weights(model, keys_to_backup)

        # Store this LoRA's diffs and initial strength (use path as key)
        state["loras"][str(lora_path)] = {
            "diffs": lora_diffs,
            "scale": strength,
        }

        # Apply all LoRAs with their current scales
        LoRAManager._apply_all_loras(model)

        if applied_count > 0:
            logger.info(f"load_adapter: Applied {applied_count} LoRA weight patches")
            logger.debug(
                f"load_adapter: Skipped {skipped_no_match} no match, "
                f"{skipped_no_down} no down weight, {skipped_no_model_key} no model key"
            )
        else:
            logger.warning(f"load_adapter: No LoRA patches applied (check model compatibility)")

        return str(lora_path)


    @staticmethod
    def load_adapters_from_list(
        model: torch.nn.Module,
        lora_configs: List[Dict[str, Any]],
        logger_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Load multiple LoRA adapters by patching model weights.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: path, scale

        Example:
            >>> loaded = LoRAManager.load_adapters_from_list(
            ...     model=pipeline.transformer,
            ...     lora_configs=[{"path": "models/lora/style.safetensors", "scale": 1.0}],
            ...     logger_prefix="MyPipeline.__init__: "
            ... )
        """
        loaded_adapters = []

        if not lora_configs:
            return loaded_adapters

        for lora_config in lora_configs:
            lora_path = lora_config.get("path")
            if not lora_path:
                logger.warning(f"{logger_prefix}Skipping LoRA config with no path specified")
                continue

            scale = lora_config.get("scale", 1.0)

            start = time.time()
            try:
                returned_path = LoRAManager.load_adapter(
                    model=model,
                    lora_path=lora_path,
                    strength=scale
                )

                elapsed = time.time() - start
                logger.info(f"{logger_prefix}Loaded LoRA '{Path(lora_path).name}' (scale={scale}) in {elapsed:.3f}s")

                loaded_adapters.append({
                    "path": returned_path,
                    "scale": scale,
                })
            except FileNotFoundError as e:
                logger.error(f"{logger_prefix}LoRA file not found: {lora_path}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. File not found: {lora_path}. "
                    f"Ensure the file exists in the models/lora/ directory."
                ) from e
            except Exception as e:
                logger.error(f"{logger_prefix}Failed to load LoRA adapter: {e}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. Pipeline cannot start without all configured LoRAs. "
                    f"Error: {e}"
                ) from e

        return loaded_adapters

    @staticmethod
    def update_adapter_scales(
        model: torch.nn.Module,
        loaded_adapters: List[Dict[str, Any]],
        scale_updates: List[Dict[str, Any]],
        logger_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Update scales for loaded LoRA adapters at runtime.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages

        Returns:
            Updated loaded_adapters list with new scale values

        Example:
            >>> self.loaded_lora_adapters = LoRAManager.update_adapter_scales(
            ...     model=self.stream.generator.model,
            ...     loaded_adapters=self.loaded_lora_adapters,
            ...     scale_updates=[{"path": "models/lora/style1.safetensors", "scale": 0.5}],
            ...     logger_prefix="Pipeline.prepare: "
            ... )
        """
        if not scale_updates:
            return loaded_adapters

        state = LoRAManager._get_model_state(model)

        # Build map of paths to new scales
        scale_map = {}
        for update in scale_updates:
            path = update.get("path")
            scale = update.get("scale")
            if path is not None and scale is not None:
                scale_map[path] = scale

        if not scale_map:
            return loaded_adapters

        # Update scales in stored LoRA info
        scales_changed = False
        for path, new_scale in scale_map.items():
            if path in state["loras"]:
                old_scale = state["loras"][path]["scale"]
                if abs(old_scale - new_scale) > 1e-6:
                    state["loras"][path]["scale"] = new_scale
                    scales_changed = True
                    logger.info(f"{logger_prefix}Updated LoRA '{Path(path).name}' scale from {old_scale} to {new_scale}")

                    # Also update in loaded_adapters list
                    for adapter_info in loaded_adapters:
                        if adapter_info["path"] == path:
                            adapter_info["scale"] = new_scale
                            break

        # Reapply all LoRAs with updated scales
        if scales_changed:
            LoRAManager._apply_all_loras(model)
            logger.debug(f"{logger_prefix}Reapplied all LoRAs with updated scales")

        return loaded_adapters
