"""Mixin for pipelines that support LoRA adapters on the WAN generator model.

This mixin is intentionally thin: it delegates all heavy lifting to the
LoRA engine so that modular block graphs remain completely unaware of LoRA.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..wan2_1.lora_engine import LoRAEngine


class LoRAEnabledPipeline:
    """Shared LoRA integration for WAN-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_loras(config, model)` during __init__ to attach adapters to
      the underlying diffusion model (typically components.generator.model).
    - Call `_handle_lora_scale_updates(lora_scales, model)` from their
      prepare/forward path to apply runtime scale changes (for strategies that
      support them, e.g. runtime_peft).

    The mixin keeps track of:
    - self._lora_merge_mode: currently active merge strategy
    - self.loaded_lora_adapters: list of {path, scale, adapter_name?}
    """

    _lora_merge_mode: str | None = None
    loaded_lora_adapters: list[dict[str, Any]] | None = None

    def _init_loras(self, config: Any, model) -> Any:
        """Initialize LoRA adapters based on config and return (possibly wrapped) model.

        Args:
            config: OmegaConf / config object used to construct the pipeline.
                    Expected to have attributes/fields:
                    - loras: list of {path, scale}
                    - lora_merge_mode: strategy string
            model:  Underlying diffusion model to which LoRA should be applied.

        Returns:
            Model instance which may be wrapped (e.g. PEFT model) depending on strategy.
        """
        # Access both attribute-style and dict-style config
        loras = (
            getattr(config, "loras", None)
            if hasattr(config, "loras")
            else config.get("loras", None)
        )  # type: ignore[arg-type]
        merge_mode = (
            getattr(config, "lora_merge_mode", None)
            if hasattr(config, "lora_merge_mode")
            else config.get("lora_merge_mode", None)  # type: ignore[arg-type]
        )

        if not loras:
            # No LoRA requested
            self.loaded_lora_adapters = []
            self._lora_merge_mode = merge_mode
            return model

        self._lora_merge_mode = merge_mode or "runtime_peft"

        wrapped_model, loaded_adapters = LoRAEngine.load_adapters_from_list(
            model=model,
            lora_configs=list(loras),
            merge_mode=self._lora_merge_mode,
            logger_prefix=f"{self.__class__.__name__}.__init__: ",
        )

        self.loaded_lora_adapters = loaded_adapters
        return wrapped_model

    def _handle_lora_scale_updates(
        self,
        lora_scales: Iterable[dict[str, Any]] | None,
        model,
    ) -> None:
        """Apply runtime scale updates for loaded LoRA adapters if supported.

        Args:
            lora_scales: Iterable of {path, scale} updates from the client.
            model:       Model instance currently used by the pipeline (may be wrapped).
        """
        if not lora_scales:
            return

        if not self.loaded_lora_adapters:
            return

        # Some strategies (e.g. permanent_merge) do not support runtime updates.
        if getattr(self, "_lora_merge_mode", None) == "permanent_merge":
            return

        updated_adapters = LoRAEngine.update_adapter_scales(
            model=model,
            loaded_adapters=list(self.loaded_lora_adapters),
            scale_updates=list(lora_scales),
            merge_mode=getattr(self, "_lora_merge_mode", None),
            logger_prefix=f"{self.__class__.__name__}.prepare: ",
        )

        self.loaded_lora_adapters = updated_adapters
