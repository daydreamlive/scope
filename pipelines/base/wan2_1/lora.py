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
        """Backup original model weights before any LoRA patching using direct parameter access."""
        state = LoRAManager._get_model_state(model)

        if state["backed_up"]:
            return  # Already backed up

        print(f"_backup_original_weights: Backing up {len(keys_to_backup)} weights using direct parameter access...")

        # Build map of parameter names to actual parameters
        param_dict = dict(model.named_parameters())

        for key in keys_to_backup:
            # named_parameters() keys already include .weight suffix, use key directly
            if key in param_dict and key not in state["original_weights"]:
                # Clone the actual parameter tensor directly (no full state_dict copy)
                state["original_weights"][key] = param_dict[key].data.clone()

        state["backed_up"] = True
        print(f"_backup_original_weights: Backed up {len(state['original_weights'])} weights")
        logger.debug(f"_backup_original_weights: Backed up {len(state['original_weights'])} original weights")

    @staticmethod
    def _apply_all_loras(model: torch.nn.Module) -> None:
        """Recompute patched weights from originals + all loaded LoRA diffs using direct parameter access."""
        state = LoRAManager._get_model_state(model)

        if not state["backed_up"]:
            return  # Nothing to apply

        print(f"_apply_all_loras: Building parameter map for direct access...")
        # Build map of parameter names to actual parameters (fast, no copying)
        param_dict = dict(model.named_parameters())
        print(f"_apply_all_loras: Got {len(param_dict)} parameters")

        # Compute and apply each weight directly (no intermediate cloning!)
        print(f"_apply_all_loras: Computing and applying {len(state['original_weights'])} weights with {len(state['loras'])} LoRA(s)...")
        updated_count = 0

        for key, original in state["original_weights"].items():
            if key not in param_dict:
                continue

            current_param = param_dict[key]

            # Start with original (no clone yet - compute in-place if possible)
            # Move original to target device/dtype first
            new_weight = original.to(device=current_param.device, dtype=current_param.dtype)

            # Add all LoRA contributions directly
            for path, lora_info in state["loras"].items():
                if key in lora_info["diffs"]:
                    scale = lora_info["scale"]
                    diff = lora_info["diffs"][key]
                    # Add scaled diff directly (converted to correct device/dtype)
                    new_weight = new_weight + (scale * diff).to(device=current_param.device, dtype=current_param.dtype)

            # Ensure tensor is contiguous for efficient downstream operations (e.g., quantization)
            if not new_weight.is_contiguous():
                new_weight = new_weight.contiguous()

            # Assign the final computed weight directly to parameter
            current_param.data = new_weight
            updated_count += 1

        print(f"_apply_all_loras: Successfully updated {updated_count} parameters")

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
        print(f"load_adapter: Starting to load {lora_path}")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"load_adapter: LoRA file not found: {lora_path}")

        state = LoRAManager._get_model_state(model)

        # Load LoRA weights
        print(f"load_adapter: Loading LoRA weights from file...")
        if lora_path.endswith('.safetensors'):
            lora_state = load_file(lora_path)
        else:
            lora_state = torch.load(lora_path, map_location='cpu')

        print(f"load_adapter: Loaded {len(lora_state)} keys from LoRA file")
        logger.debug(f"load_adapter: Loaded {len(lora_state)} keys from {lora_path}")

        print(f"load_adapter: Building parameter map (fast, no full copy)...")
        # Build map from named_parameters instead of state_dict (much faster!)
        param_dict = dict(model.named_parameters())
        print(f"load_adapter: Got {len(param_dict)} parameters")
        if param_dict:
            example_param_name = next(iter(param_dict.keys()))
            print(f"load_adapter: Example parameter name: {example_param_name}")

        # Create a lightweight "fake" state dict with just keys (no actual tensors)
        # named_parameters() already includes .weight/.bias suffixes, so use them directly
        print(f"load_adapter: Building key map for LoRA matching...")
        fake_state_dict = {name: None for name in param_dict.keys()}
        key_map = LoRAManager._build_key_map(fake_state_dict)

        print(f"load_adapter: Built key map with {len(key_map)} entries")
        logger.debug(f"load_adapter: Built key map with {len(key_map)} entries")

        # First pass: collect all affected keys and compute diffs
        lora_diffs = {}  # {model_key: diff_tensor}
        keys_to_backup = []
        applied_count = 0
        skipped_no_match = 0
        skipped_no_down = 0
        skipped_no_model_key = 0

        print(f"load_adapter: Processing {len(lora_state)} LoRA keys...")
        first_lora_up_key = None
        for lora_key in lora_state.keys():
            if first_lora_up_key is None and ('.lora_up.weight' in lora_key or '.lora_B.weight' in lora_key):
                first_lora_up_key = lora_key
        if first_lora_up_key:
            print(f"load_adapter: Example LoRA key: {first_lora_up_key}")

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

            if model_key is None:
                skipped_no_model_key += 1
                if skipped_no_model_key <= 3:
                    print(f"load_adapter: DEBUG - No model key found for base_key={base_key}, normalized={normalized_key}")
                    logger.debug(f"load_adapter: No model key found for base_key={base_key}, normalized={normalized_key}")
                continue

            # Get the actual parameter directly using model_key
            # (named_parameters already includes .weight suffix, so model_key should match)
            if model_key not in param_dict:
                skipped_no_model_key += 1
                if skipped_no_model_key <= 3:
                    print(f"load_adapter: DEBUG - Parameter {model_key} not found in param_dict")
                    logger.debug(f"load_adapter: Parameter {model_key} not found in model")
                continue

            # Get original weight directly from parameter (no state_dict copy!)
            original_weight = param_dict[model_key].data

            # Get alpha if present
            alpha = lora_state.get(alpha_key)
            if alpha is not None:
                alpha = alpha.item()

            # Compute the LoRA diff (without strength, we'll apply that later)
            lora_up = lora_state[lora_key]
            lora_down = lora_state[down_key]

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

            if applied_count <= 3:
                print(f"load_adapter: DEBUG - Successfully matched: lora_key={base_key} -> model_key={model_key}")

        print(f"load_adapter: Finished processing LoRA keys. Applied {applied_count} patches.")
        print(f"load_adapter: Skipped: no_match={skipped_no_match}, no_down={skipped_no_down}, no_model_key={skipped_no_model_key}")

        # Backup original weights before first LoRA
        if keys_to_backup:
            print(f"load_adapter: Backing up {len(keys_to_backup)} original weights...")
            LoRAManager._backup_original_weights(model, keys_to_backup)
            print(f"load_adapter: Backup complete")

        # Store this LoRA's diffs and initial strength (use path as key)
        state["loras"][str(lora_path)] = {
            "diffs": lora_diffs,
            "scale": strength,
        }

        # Apply all LoRAs with their current scales
        print(f"load_adapter: Applying all LoRAs to model weights...")
        LoRAManager._apply_all_loras(model)
        print(f"load_adapter: LoRA application complete")

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
            print(f"load_adapters_from_list: No LoRA configs provided")
            return loaded_adapters

        print(f"load_adapters_from_list: Loading {len(lora_configs)} LoRA adapter(s)...")
        for idx, lora_config in enumerate(lora_configs):
            print(f"load_adapters_from_list: Processing LoRA {idx + 1}/{len(lora_configs)}")
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
