"""
Permanent merge LoRA manager for WAN models - maximum inference performance.

This implementation merges LoRA weights directly into model weights at load time,
providing zero inference overhead. No runtime scale updates are supported - the
scale is fixed at load time. Ideal for production deployment where LoRA scale
is predetermined and maximum FPS is critical.
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch

from pipelines.wan2_1.lora.utils import (
    build_key_map,
    calculate_lora_scale,
    find_lora_pair,
    load_lora_weights,
    normalize_lora_key,
)

logger = logging.getLogger(__name__)

__all__ = ["PermanentMergeLoRAStrategy"]


class PermanentMergeLoRAStrategy:
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
            >>> from pipelines.wan2_1.lora.strategies.permanent_merge_lora import PermanentMergeLoRAStrategy
            >>> path = PermanentMergeLoRAStrategy.load_adapter(
            ...     model=pipeline.transformer,
            ...     lora_path="models/lora/my-style.safetensors",
            ...     strength=1.0
            ... )
        """
        # Load LoRA weights
        lora_state = load_lora_weights(lora_path)
        logger.debug(f"load_adapter: Loaded {len(lora_state)} keys from {lora_path}")

        model_state = model.state_dict()
        key_map = build_key_map(model_state)
        logger.debug(f"load_adapter: Built key map with {len(key_map)} entries")

        # Compute merged weights
        merged_weights = {}
        applied_count = 0
        skipped_no_match = 0
        skipped_no_model_key = 0
        processed_keys = set()

        for lora_key in lora_state.keys():
            if lora_key in processed_keys:
                continue

            # Find LoRA pair using shared utility
            pair_result = find_lora_pair(lora_key, lora_state)
            if pair_result is None:
                skipped_no_match += 1
                continue

            base_key, alpha_key, lora_A, lora_B = pair_result

            # Mark both keys as processed
            if ".lora_up.weight" in lora_key:
                processed_keys.add(lora_key)
                processed_keys.add(f"{base_key}.lora_down.weight")
            elif ".lora_B.weight" in lora_key:
                processed_keys.add(lora_key)
                processed_keys.add(f"{base_key}.lora_A.weight")

            # Normalize the base key to match model format
            normalized_key = normalize_lora_key(base_key)

            # Find the model weight key
            model_key = key_map.get(normalized_key)
            if model_key is None:
                model_key = key_map.get(f"diffusion_model.{normalized_key}")

            if model_key is None or model_key not in model_state:
                skipped_no_model_key += 1
                if skipped_no_model_key <= 3:
                    logger.debug(
                        f"load_adapter: No model key found for base_key={base_key}, normalized={normalized_key}"
                    )
                continue

            # Extract alpha
            alpha = None
            if alpha_key and alpha_key in lora_state:
                alpha = lora_state[alpha_key].item()

            # Compute the LoRA contribution
            original_weight = model_state[model_key]

            # Move LoRA weights to same device as model weight
            lora_A = lora_A.to(device=original_weight.device)
            lora_B = lora_B.to(device=original_weight.device)

            # Compute scale: alpha / rank
            rank = lora_A.shape[0]
            scale = calculate_lora_scale(alpha, rank)

            # Compute LoRA diff: B @ A (note: lora_B is up, lora_A is down)
            lora_diff = torch.mm(
                lora_B.float().flatten(start_dim=1), lora_A.float().flatten(start_dim=1)
            ).reshape(original_weight.shape)

            # Merge: W_final = W_base + strength * scale * diff
            # Convert to original weight's dtype before merging
            scaled_diff = (strength * scale * lora_diff).to(
                dtype=original_weight.dtype, device=original_weight.device
            )
            merged_weight = original_weight + scaled_diff

            merged_weights[model_key] = merged_weight
            applied_count += 1

        # Apply merged weights to model using load_state_dict
        # This handles quantized tensors properly
        if merged_weights:
            model.load_state_dict(merged_weights, strict=False)
            logger.info(
                f"load_adapter: Permanently merged {applied_count} LoRA weight patches (strength={strength})"
            )
            logger.debug(
                f"load_adapter: Skipped {skipped_no_match} no match, "
                f"{skipped_no_model_key} no model key"
            )
        else:
            logger.warning(
                "load_adapter: No LoRA patches applied (check model compatibility)"
            )

        return str(lora_path)

    @staticmethod
    def load_adapters_from_list(
        model: torch.nn.Module,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters by permanently merging into model weights.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: path, scale

        Example:
            >>> loaded = PermanentMergeLoRAStrategy.load_adapters_from_list(
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
                logger.warning(
                    f"{logger_prefix}Skipping LoRA config with no path specified"
                )
                continue

            scale = lora_config.get("scale", 1.0)

            start = time.time()
            try:
                returned_path = PermanentMergeLoRAStrategy.load_adapter(
                    model=model, lora_path=lora_path, strength=scale
                )

                elapsed = time.time() - start
                logger.info(
                    f"{logger_prefix}Permanently merged LoRA '{Path(lora_path).name}' (scale={scale}) in {elapsed:.3f}s"
                )

                loaded_adapters.append(
                    {
                        "path": returned_path,
                        "scale": scale,
                    }
                )
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
        if scale_updates:
            logger.warning(
                f"{logger_prefix}Runtime LoRA scale updates are NOT SUPPORTED in permanent merge mode. "
                f"LoRA weights are permanently merged at load time. To change scales, reload the model "
                f"with different scale values in the lora configs."
            )

        return loaded_adapters
