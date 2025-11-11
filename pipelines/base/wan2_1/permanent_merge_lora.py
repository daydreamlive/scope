"""
Permanent merge LoRA manager for WAN models - maximum inference performance.

This implementation merges LoRA weights directly into model weights at load time,
providing zero inference overhead. No runtime scale updates are supported - the
scale is fixed at load time. Ideal for production deployment where LoRA scale
is predetermined and maximum FPS is critical.
"""
from typing import Dict, Any, List
import os
import time
import logging
from pathlib import Path
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

__all__ = ["PermanentMergeLoRAManager"]


class PermanentMergeLoRAManager:
    """
    Manages LoRA adapters via permanent weight merging at load time.

    This merges LoRA weights directly: W_final = W_base + scale * (lora_B @ lora_A)
    The LoRA matrices and original weights are discarded after merging, minimizing
    memory usage and providing zero inference overhead.

    Trade-off: Runtime scale updates are not supported. The scale is permanently
    baked into the weights at load time.

    Compatible with FP8 quantization - merge happens before or after quantization
    depending on when LoRAs are loaded.
    """

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
    def load_adapter(
        model: torch.nn.Module,
        lora_path: str,
        strength: float = 1.0,
    ) -> str:
        """
        Load LoRA by permanently merging into model weights.

        This performs one-time weight merging: W_final = W_base + strength * (lora_B @ lora_A)
        After merging, the LoRA matrices are discarded. No runtime scale updates are supported.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Strength multiplier for LoRA effect (permanently baked in)

        Returns:
            The lora_path (used as identifier)

        Raises:
            FileNotFoundError: If the LoRA file does not exist

        Example:
            >>> from pipelines.base.wan2_1.permanent_merge_lora import PermanentMergeLoRAManager
            >>> path = PermanentMergeLoRAManager.load_adapter(
            ...     model=pipeline.transformer,
            ...     lora_path="models/lora/my-style.safetensors",
            ...     strength=1.0
            ... )
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"load_adapter: LoRA file not found: {lora_path}")

        # Load LoRA weights
        if lora_path.endswith('.safetensors'):
            lora_state = load_file(lora_path)
        else:
            lora_state = torch.load(lora_path, map_location='cpu')

        logger.debug(f"load_adapter: Loaded {len(lora_state)} keys from {lora_path}")

        model_state = model.state_dict()
        key_map = PermanentMergeLoRAManager._build_key_map(model_state)
        logger.debug(f"load_adapter: Built key map with {len(key_map)} entries")

        # Compute merged weights
        merged_weights = {}
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
            normalized_key = PermanentMergeLoRAManager._normalize_lora_key(base_key)

            # Find the model weight key
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

            # Compute the LoRA contribution
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

            # Compute LoRA diff: up @ down
            lora_diff = torch.mm(
                lora_up.float().flatten(start_dim=1),
                lora_down.float().flatten(start_dim=1)
            ).reshape(original_weight.shape)

            # Merge: W_final = W_base + strength * scale * diff
            # Convert to original weight's dtype before merging
            scaled_diff = (strength * scale * lora_diff).to(dtype=original_weight.dtype, device=original_weight.device)
            merged_weight = original_weight + scaled_diff

            merged_weights[model_key] = merged_weight
            applied_count += 1

        # Apply merged weights to model using load_state_dict
        # This handles quantized tensors properly
        if merged_weights:
            model.load_state_dict(merged_weights, strict=False)
            logger.info(f"load_adapter: Permanently merged {applied_count} LoRA weight patches (strength={strength})")
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
        Load multiple LoRA adapters by permanently merging into model weights.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: path, scale

        Example:
            >>> loaded = PermanentMergeLoRAManager.load_adapters_from_list(
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
                returned_path = PermanentMergeLoRAManager.load_adapter(
                    model=model,
                    lora_path=lora_path,
                    strength=scale
                )

                elapsed = time.time() - start
                logger.info(f"{logger_prefix}Permanently merged LoRA '{Path(lora_path).name}' (scale={scale}) in {elapsed:.3f}s")

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
        if scale_updates:
            logger.warning(
                f"{logger_prefix}Runtime LoRA scale updates are NOT SUPPORTED in permanent merge mode. "
                f"LoRA weights are permanently merged at load time. To change scales, reload the model "
                f"with different scale values in the lora configs."
            )

        return loaded_adapters
