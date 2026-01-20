"""
LoRA utilities for WAN models - thin wrapper that delegates to strategy implementations.

This module provides a unified interface for different LoRA merge strategies:
- permanent_merge: Maximum FPS, no runtime updates (permanent_merge_lora.py)
- runtime_peft: Instant updates with per-frame overhead (peft_lora.py)

Supports local .safetensors and .bin files from models/lora/ directory.
"""

import logging
from collections import defaultdict
from typing import Any

from .strategies.module_targeted_lora import (
    ModuleTargetedLoRAStrategy,
)
from .strategies.peft_lora import PeftLoRAStrategy
from .strategies.permanent_merge_lora import (
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
      - Uses PEFT wrapping without runtime scale updates
    """

    # Default strategy if none specified
    DEFAULT_STRATEGY = "permanent_merge"

    # Order in which modes should be loaded (important: permanent_merge must be first)
    MODE_LOAD_ORDER = ["permanent_merge", "runtime_peft", "module_targeted"]

    @staticmethod
    def _resolve_mode(mode: str | None, fallback: str | None = None) -> str:
        """Resolve merge mode with fallback chain: mode -> fallback -> DEFAULT_STRATEGY."""
        return mode or fallback or LoRAManager.DEFAULT_STRATEGY

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
    def _group_by_mode(
        items: list[dict[str, Any]],
        mode_extractor: callable,
        fallback_mode: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Group items by merge_mode using custom extraction logic.

        Args:
            items: List of items to group
            mode_extractor: Function that extracts mode from an item (returns str | None)
            fallback_mode: Fallback mode if extractor returns None

        Returns:
            Dictionary mapping merge_mode to list of items
        """
        grouped = defaultdict(list)
        for item in items:
            mode = mode_extractor(item)
            resolved = LoRAManager._resolve_mode(mode, fallback_mode)
            grouped[resolved].append(item)
        return dict(grouped)

    @staticmethod
    def _load_loras_for_mode(
        mode: str,
        loras: list[dict[str, Any]],
        model,
        logger_prefix: str,
        target_modules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load LoRAs for a specific mode and inject merge_mode into adapter info.

        Args:
            mode: Merge strategy name (permanent_merge, runtime_peft, module_targeted)
            loras: List of LoRA configs to load with this strategy
            model: PyTorch model
            logger_prefix: Prefix for log messages
            target_modules: For module_targeted mode, list of module class names to target

        Returns:
            List of loaded adapter info dicts with merge_mode injected
        """
        logger.info(f"{logger_prefix}Loading {len(loras)} LoRA(s) with {mode} strategy")
        manager_class = LoRAManager._get_manager_class(mode)

        if mode == "module_targeted" and target_modules is not None:
            loaded = manager_class.load_adapters_from_list(
                model, loras, logger_prefix, target_modules=target_modules
            )
        else:
            loaded = manager_class.load_adapters_from_list(model, loras, logger_prefix)

        for adapter_info in loaded:
            adapter_info["merge_mode"] = mode

        return loaded

    @staticmethod
    def load_adapters_from_list(
        model,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
        merge_mode: str = None,
        target_modules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters, supporting per-LoRA merge strategies.

        Each LoRA config can optionally specify its own merge_mode. If not specified,
        uses the default merge_mode parameter. LoRAs are grouped by strategy and loaded
        in order: permanent_merge first (merged and unloaded), then runtime_peft.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys:
                - path (str, required): LoRA file path
                - scale (float, optional, default=1.0): LoRA strength
                - merge_mode (str, optional): Per-LoRA merge strategy
            logger_prefix: Prefix for log messages
            merge_mode: Default strategy to use if not specified in LoRA config
            target_modules: For module_targeted mode, list of module class names to target

        Returns:
            List of loaded adapter info dicts with keys: path, scale, merge_mode
        """
        if not lora_configs:
            return []

        default_merge_mode = LoRAManager._resolve_mode(merge_mode)
        loras_by_mode = LoRAManager._group_by_mode(
            lora_configs,
            lambda cfg: cfg.get("merge_mode"),
            default_merge_mode,
        )

        all_loaded_adapters = []

        # Load LoRAs in order defined by MODE_LOAD_ORDER
        for mode in LoRAManager.MODE_LOAD_ORDER:
            if mode not in loras_by_mode:
                continue

            loaded = LoRAManager._load_loras_for_mode(
                mode,
                loras_by_mode[mode],
                model,
                logger_prefix,
                target_modules=target_modules,
            )
            all_loaded_adapters.extend(loaded)

        return all_loaded_adapters

    @staticmethod
    def _build_adapter_mode_map(
        loaded_adapters: list[dict[str, Any]],
        fallback_mode: str | None,
    ) -> dict[str, str]:
        """Build path to merge_mode lookup map from loaded adapters.

        Args:
            loaded_adapters: List of loaded adapter info dicts
            fallback_mode: Fallback merge_mode if not specified in adapter

        Returns:
            Dictionary mapping adapter path to merge_mode
        """
        adapter_path_to_mode: dict[str, str] = {}

        for adapter in loaded_adapters:
            adapter_path = adapter.get("path")
            if adapter_path:
                adapter_mode = LoRAManager._resolve_mode(
                    adapter.get("merge_mode"), fallback_mode
                )
                adapter_path_to_mode[adapter_path] = adapter_mode

        return adapter_path_to_mode

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

        Supports per-LoRA merge modes by looking up the merge_mode from loaded_adapters.
        If a LoRA's merge_mode is not found in loaded_adapters, falls back to the
        provided merge_mode parameter.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts (with merge_mode)
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages
            merge_mode: Fallback strategy if merge_mode not found in loaded_adapters

        Returns:
            Updated loaded_adapters list with new scale values
        """
        if not scale_updates:
            return loaded_adapters

        # Build lookup map and group updates by merge_mode
        adapter_mode_map = LoRAManager._build_adapter_mode_map(
            loaded_adapters, merge_mode
        )
        updates_by_mode = LoRAManager._group_by_mode(
            scale_updates,
            lambda upd: adapter_mode_map.get(upd.get("path")),
            merge_mode,
        )

        # Update adapters for each merge_mode
        updated_adapters = list(loaded_adapters)
        for mode, updates in updates_by_mode.items():
            manager_class = LoRAManager._get_manager_class(mode)
            updated_adapters = manager_class.update_adapter_scales(
                model, updated_adapters, updates, logger_prefix
            )

        return updated_adapters
