"""Mixin for pipelines that support LoRA adapters on the WAN generator model.

This mixin is intentionally thin: it delegates all heavy lifting to the
LoRA managers so that modular block graphs remain completely unaware of LoRA.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pipelines.wan2_1.lora import LoRAManager
from pipelines.wan2_1.lora.strategies.cuda_graph_recapture_lora import (
    CudaGraphRecaptureLoRAManager,
)


class LoRAEnabledPipeline:
    """Shared LoRA integration for WAN-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_loras(config, model)` during __init__ to attach adapters to
      the underlying diffusion model (typically components.generator.model).
    - Call `_handle_lora_scale_updates(lora_scales, model)` from their
      prepare/forward path to apply runtime scale changes (for strategies that
      support them, e.g. runtime_peft).

    Supports four LoRA implementations underneath LoRAManager:
    - permanent_merge: one-time merge at load (zero overhead, no runtime updates)
    - runtime_peft: runtime LoRA application (<1s updates, FPS overhead)
    - gpu_reconstruct: GPU reconstruction (slow updates, high FPS)
    - cuda_graph_recapture: CUDA Graph + PEFT (fast FPS, 1–5s updates)

    The mixin keeps track of:
    - self._lora_merge_mode: currently active merge strategy
    - self.loaded_lora_adapters: list of {path, scale, adapter_name?}
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
        # Access both attribute-style and mapping-style config
        if hasattr(config, "get"):
            lora_configs = config.get("loras", [])  # OmegaConf / dict
            lora_merge_mode = config.get("lora_merge_mode", None)
        else:
            lora_configs = getattr(config, "loras", []) or []
            lora_merge_mode = getattr(config, "lora_merge_mode", None)

        # Handle legacy use_peft_lora boolean if present on config
        if hasattr(config, "get"):
            use_peft_flag = config.get("use_peft_lora", None)
        else:
            use_peft_flag = getattr(config, "use_peft_lora", None)

        if lora_merge_mode is None and use_peft_flag is not None:
            lora_merge_mode = "runtime_peft" if use_peft_flag else "gpu_reconstruct"

        # Default to gpu_reconstruct if still unset (matches original branch)
        lora_merge_mode = lora_merge_mode or "gpu_reconstruct"

        print(f"_init_loras: Found {len(lora_configs)} LoRA configs to load")
        print(f"_init_loras: Using merge mode: {lora_merge_mode}")

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
        )

        print(
            f"_init_loras: Completed, loaded {len(self.loaded_lora_adapters)} LoRA adapters"
        )

        # For CUDA Graph Re-capture mode, return the PEFT-wrapped model
        if lora_merge_mode == "cuda_graph_recapture":
            wrapped_model = CudaGraphRecaptureLoRAManager.get_wrapped_model(model)
            if wrapped_model is not None:
                print(
                    "_init_loras: Returning PEFT-wrapped model for CUDA Graph capture"
                )
                return wrapped_model
            msg = (
                "_init_loras: Failed to get wrapped model from CUDA Graph manager. "
                "This is a critical error for cuda_graph_recapture mode."
            )
            raise RuntimeError(msg)

        # For other modes, return the original model
        return model

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
        merge_mode = getattr(self, "_lora_merge_mode", None)
        if merge_mode == "permanent_merge":
            # Manager will already log a warning, but short-circuit here as well.
            return

        updated_adapters = LoRAManager.update_adapter_scales(
            model=model,
            loaded_adapters=list(self.loaded_lora_adapters),
            scale_updates=list(lora_scales),
            logger_prefix=f"{self.__class__.__name__}.prepare: ",
            merge_mode=merge_mode,
        )

        self.loaded_lora_adapters = updated_adapters
