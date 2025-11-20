"""
Permanent merge LoRA strategy for WAN models.

Merges LoRA weights into model weights at load time using PEFT's merge_and_unload(),
providing zero inference overhead. LoRA scales are fixed at load time and cannot be
updated at runtime. Ideal for production deployment where maximum FPS is critical.
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch

from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAStrategy

logger = logging.getLogger(__name__)

__all__ = ["PermanentMergeLoRAStrategy"]


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

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Strength multiplier for LoRA effect

        Returns:
            The lora_path (used as identifier)

        Raises:
            FileNotFoundError: If the LoRA file does not exist
        """
        start_time = time.time()
        logger.info(
            f"load_adapter: Loading and permanently merging LoRA from {lora_path} (strength={strength})"
        )

        # Use PeftLoRAStrategy to load the adapter (wraps model with PEFT)
        adapter_name = PeftLoRAStrategy.load_adapter(
            model=model, lora_path=lora_path, strength=strength
        )

        # Get the PEFT-wrapped model from the cache
        peft_model = PeftLoRAStrategy._get_peft_model(model)
        if peft_model is None:
            raise RuntimeError(
                f"load_adapter: Failed to get PEFT model after loading adapter '{adapter_name}'"
            )

        # Get the actual PEFT model (unwrap torch.compile if needed)
        target_model = (
            peft_model._orig_mod if hasattr(peft_model, "_orig_mod") else peft_model
        )

        # Merge the adapter weights into base model and unwrap
        merged_model = target_model.merge_and_unload(
            safe_merge=True, adapter_names=[adapter_name]
        )

        # Copy merged weights back to original model
        # This is necessary because merge_and_unload returns a new model instance
        model.load_state_dict(merged_model.state_dict(), strict=True)

        # Clean up PEFT model from cache
        if model in PeftLoRAStrategy._peft_models:
            del PeftLoRAStrategy._peft_models[model]

        elapsed = time.time() - start_time
        logger.info(
            f"load_adapter: Permanently merged LoRA '{Path(lora_path).name}' in {elapsed:.3f}s"
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
