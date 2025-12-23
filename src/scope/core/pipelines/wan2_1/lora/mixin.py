"""Mixin for pipelines that support LoRA adapters on the WAN generator model.

This mixin is intentionally thin: it delegates all heavy lifting to the
LoRA managers so that modular block graphs remain completely unaware of LoRA.

Note: This LoRA integration is specifically designed for the Wan2.1 architecture
and may not be compatible with other model architectures without modification.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from .manager import LoRAManager

logger = logging.getLogger(__name__)

PERMANENT_MERGE_MODE = "permanent_merge"
RUNTIME_PEFT_MODE = "runtime_peft"


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get value from config whether it's dict-like or object-like."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class LoRAEnabledPipeline:
    """Shared LoRA integration for WAN-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_loras(config, model)` during __init__ to attach adapters to
      the underlying diffusion model (typically components.generator.model).
    - Call `_handle_lora_scale_updates(lora_scales, model)` from their
      prepare/forward path to apply runtime scale changes (for strategies that
      support them, e.g. runtime_peft).

    Supports two LoRA implementations underneath LoRAManager:
    - permanent_merge: one-time merge at load (zero overhead, no runtime updates)
    - runtime_peft: runtime LoRA application (<1s updates, FPS overhead)

    The mixin keeps track of:
    - self._lora_merge_mode: currently active merge strategy
    - self.loaded_lora_adapters: list of {path, scale, adapter_name?}

    Architecture Compatibility:
    This implementation is specifically designed for Wan2.1 architecture models
    and targets their specific Linear layer structure. Using this with other
    architectures may require modifications to the LoRA strategies.
    """

    _lora_merge_mode: str | None = None
    loaded_lora_adapters: list[dict[str, Any]] | None = None

    def _init_loras(self, config: Any, model) -> Any:
        """Initialize LoRA adapters based on config and return (possibly wrapped) model.

        Args:
            config: Pipeline configuration / OmegaConf object that may contain:
                - 'loras': List of LoRA configs ({path, scale, ...})
                - 'lora_merge_mode': Strategy string
            model:  Underlying diffusion model to which LoRA should be applied.

        Returns:
            Model instance which may be wrapped (e.g. PEFT model) depending on strategy.
        """
        lora_configs = _get_config_value(config, "loras", []) or []
        lora_merge_mode = _get_config_value(
            config, "_lora_merge_mode"
        ) or _get_config_value(config, "lora_merge_mode")

        # Get target modules for module_targeted mode
        target_modules = None
        if lora_merge_mode == "module_targeted":
            target_modules = _get_config_value(config, "_lora_target_modules")

        # Handle legacy use_peft_lora boolean if present on config
        use_peft_flag = _get_config_value(config, "use_peft_lora")
        if lora_merge_mode is None and use_peft_flag is not None:
            lora_merge_mode = (
                RUNTIME_PEFT_MODE if use_peft_flag else PERMANENT_MERGE_MODE
            )

        # Default to permanent_merge if still unset
        lora_merge_mode = lora_merge_mode or PERMANENT_MERGE_MODE

        logger.info("_init_loras: Found %d LoRA configs to load", len(lora_configs))
        logger.debug(f"_init_loras: Using merge mode: {lora_merge_mode}")

        self._lora_merge_mode = lora_merge_mode

        if not lora_configs:
            # No LoRA requested
            self.loaded_lora_adapters = []
            return model

        # Delegate to strategy managers via LoRAManager
        self.loaded_lora_adapters = LoRAManager.load_adapters_from_list(
            model=model,
            lora_configs=list(lora_configs),
            logger_prefix=f"{self.__class__.__name__}.__init__: ",
            merge_mode=self._lora_merge_mode,
            target_modules=target_modules,
        )

        logger.info(
            "_init_loras: Completed, loaded %d LoRA adapters",
            len(self.loaded_lora_adapters),
        )

        # Return the original model (PEFT wrapping is handled internally by runtime_peft strategy)
        return model

    def _handle_lora_scale_updates(
        self,
        lora_scales: Iterable[dict[str, Any]] | None,
        model,
    ) -> None:
        """Apply runtime scale updates for loaded LoRA adapters if supported.

        Supports per-LoRA merge modes. The manager will look up the merge_mode
        for each LoRA from loaded_adapters and route updates to the appropriate
        strategy. Strategies that don't support runtime updates will handle this
        gracefully (e.g. permanent_merge logs a warning).

        Args:
            lora_scales: Iterable of {path, scale} updates from the client.
            model:       Model instance currently used by the pipeline (may be wrapped).
        """
        if not lora_scales:
            return

        if not self.loaded_lora_adapters:
            return

        scale_updates_list = list(lora_scales)
        if not scale_updates_list:
            return

        # Delegate to manager, which routes updates to appropriate strategies
        # Pass None for merge_mode so manager looks it up from loaded_adapters
        updated_adapters = LoRAManager.update_adapter_scales(
            model=model,
            loaded_adapters=list(self.loaded_lora_adapters),
            scale_updates=scale_updates_list,
            logger_prefix=f"{self.__class__.__name__}.prepare: ",
            merge_mode=None,  # Manager will look up per-LoRA merge_mode from loaded_adapters
        )

        self.loaded_lora_adapters = updated_adapters
