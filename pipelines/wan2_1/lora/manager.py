"""
LoRA utilities for WAN models - thin wrapper that delegates to strategy implementations.

This module provides a unified interface for different LoRA merge strategies:
- permanent_merge: Maximum FPS, no runtime updates (permanent_merge_lora.py)
- runtime_peft: Instant updates with per-frame overhead (peft_lora.py)

Supports local .safetensors and .bin files from models/lora/ directory.
"""

import logging
from typing import Any

from pipelines.wan2_1.lora.strategies.module_targeted_lora import (
    ModuleTargetedLoRAStrategy,
)
from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAStrategy
from pipelines.wan2_1.lora.strategies.permanent_merge_lora import (
    PermanentMergeLoRAStrategy,
)

__all__ = ["LoRAManager"]

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Unified interface for LoRA management with multiple strategies.

    Delegates to the appropriate strategy implementation based on merge_mode.

    Available strategies:
    - permanent_merge: Merges LoRA weights permanently at load time
      + Maximum inference performance (zero overhead)
      - No runtime scale updates

    - runtime_peft: Uses PEFT LoraLayer for runtime application
      + Instant scale updates (<1s)
      - ~50% inference overhead per frame

    - module_targeted: Targets specific module types (like LongLive)
      + Compatible with existing module-driven LoRA files
      - Uses PEFT wrapping with runtime scale updates
    """

    # Default strategy if none specified
    DEFAULT_STRATEGY = "permanent_merge"

    @staticmethod
    def _get_manager_class(merge_mode: str = None):
        """Get the appropriate manager class based on merge mode."""
        if merge_mode is None:
            merge_mode = LoRAManager.DEFAULT_STRATEGY

        if merge_mode == "permanent_merge":
            return PermanentMergeLoRAStrategy
        elif merge_mode == "runtime_peft":
            return PeftLoRAStrategy
        elif merge_mode == "module_targeted":
            return ModuleTargetedLoRAStrategy
        else:
            raise ValueError(
                f"Unknown merge_mode: {merge_mode}. "
                f"Supported modes: permanent_merge, runtime_peft, module_targeted"
            )

    @staticmethod
    def load_adapter(
        model, lora_path: str, strength: float = 1.0, merge_mode: str = None
    ) -> str:
        """
        Load LoRA adapter using the specified merge strategy.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier for LoRA effect (default 1.0)
            merge_mode: Strategy to use (permanent_merge, runtime_peft, module_targeted)

        Returns:
            The lora_path (used as identifier)
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)
        return manager_class.load_adapter(model, lora_path, strength)

    @staticmethod
    def load_adapters_from_list(
        model,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
        merge_mode: str = None,
        target_modules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters using the specified merge strategy.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages
            merge_mode: Strategy to use (permanent_merge, runtime_peft, module_targeted)
            target_modules: For module_targeted mode, list of module class names to target

        Returns:
            List of loaded adapter info dicts with keys: path, scale
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)

        # For module_targeted mode, pass target_modules if available
        if merge_mode == "module_targeted":
            return manager_class.load_adapters_from_list(
                model, lora_configs, logger_prefix, target_modules=target_modules
            )
        else:
            return manager_class.load_adapters_from_list(
                model, lora_configs, logger_prefix
            )

    @staticmethod
    def update_adapter_scales(
        model,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: list[dict[str, Any]],
        logger_prefix: str = "",
        merge_mode: str = None,
    ) -> list[dict[str, Any]]:
        """
        Update scales for loaded LoRA adapters at runtime.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages
            merge_mode: Strategy to use (must match the one used for loading)

        Returns:
            Updated loaded_adapters list with new scale values
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)
        return manager_class.update_adapter_scales(
            model, loaded_adapters, scale_updates, logger_prefix
        )
