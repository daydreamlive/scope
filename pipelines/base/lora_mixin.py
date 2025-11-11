"""Mixin class for pipelines that support LoRA adapters."""

from pipelines.base.wan2_1.cuda_graph_recapture_lora import (
    CudaGraphRecaptureLoRAManager,
)
from pipelines.base.wan2_1.lora import LoRAManager
from pipelines.base.wan2_1.peft_lora import PeftLoRAManager
from pipelines.base.wan2_1.permanent_merge_lora import PermanentMergeLoRAManager


class LoRAEnabledPipeline:
    """
    Mixin for pipelines that support LoRA adapters.

    Provides common functionality for loading and updating LoRA adapters
    to avoid code duplication across WAN-based pipelines.

    Supports four LoRA implementations:
    - PermanentMergeLoRAManager: One-time merge at load (zero overhead, no runtime updates)
    - PeftLoRAManager: Runtime LoRA application (<1s updates, ~50% overhead)
    - LoRAManager: GPU reconstruction approach (60s updates, no overhead)
    - CudaGraphRecaptureLoRAManager: CUDA Graph optimization (~9-10 FPS, 1-5s updates)

    Use 'lora_merge_mode' in config to choose: "permanent_merge", "runtime_peft", "gpu_reconstruct", or "cuda_graph_recapture".
    """

    def _init_loras(self, config: dict, model):
        """
        Initialize LoRA adapters during pipeline construction.

        Args:
            config: Pipeline configuration dict that may contain:
                - 'loras': List of LoRA configs
                - 'lora_merge_mode': str - One of "runtime_peft", "gpu_reconstruct" (default), "permanent_merge", "cuda_graph_recapture"
            model: The model to apply LoRAs to (typically generator.model)

        Returns:
            The model, potentially wrapped (e.g., with PEFT for cuda_graph_recapture mode).
            Caller should replace their model reference with the returned value.
        """
        lora_configs = config.get("loras", [])

        # Support both old 'use_peft_lora' boolean and new 'lora_merge_mode' string
        # Default to gpu_reconstruct
        lora_merge_mode = config.get("lora_merge_mode", "gpu_reconstruct")

        # Handle legacy use_peft_lora boolean
        if "use_peft_lora" in config and "lora_merge_mode" not in config:
            use_peft = config.get("use_peft_lora", True)
            lora_merge_mode = "runtime_peft" if use_peft else "gpu_reconstruct"

        print(f"_init_loras: Found {len(lora_configs)} LoRA configs to load")
        print(f"_init_loras: Using merge mode: {lora_merge_mode}")

        # Store which manager we're using
        self._lora_merge_mode = lora_merge_mode

        # Select manager based on mode
        if lora_merge_mode == "permanent_merge":
            manager = PermanentMergeLoRAManager
        elif lora_merge_mode == "runtime_peft":
            manager = PeftLoRAManager
        elif lora_merge_mode == "gpu_reconstruct":
            manager = LoRAManager
        elif lora_merge_mode == "cuda_graph_recapture":
            manager = CudaGraphRecaptureLoRAManager
        else:
            # Default to gpu_reconstruct
            print(
                f"_init_loras: Unknown lora_merge_mode '{lora_merge_mode}', defaulting to gpu_reconstruct"
            )
            manager = LoRAManager
            self._lora_merge_mode = "gpu_reconstruct"

        self.loaded_lora_adapters = manager.load_adapters_from_list(
            model=model,
            lora_configs=lora_configs,
            logger_prefix=f"{self.__class__.__name__}.__init__: ",
        )

        print(
            f"_init_loras: Completed, loaded {len(self.loaded_lora_adapters)} LoRA adapters"
        )

        # For CUDA Graph Re-capture mode, return the PEFT-wrapped model
        if lora_merge_mode == "cuda_graph_recapture" and hasattr(
            manager, "get_wrapped_model"
        ):
            wrapped_model = manager.get_wrapped_model(model)
            if wrapped_model is not None:
                print(
                    "_init_loras: Returning PEFT-wrapped model for CUDA Graph capture"
                )
                return wrapped_model
            else:
                raise RuntimeError(
                    "_init_loras: Failed to get wrapped model from CUDA Graph manager. "
                    "This is a critical error for cuda_graph_recapture mode."
                )

        # For other modes, return the original model
        return model

    def _handle_lora_scale_updates(self, kwargs: dict, model) -> None:
        """
        Handle runtime LoRA scale updates in prepare().

        Args:
            kwargs: Keyword arguments from prepare() that may contain 'lora_scales'
            model: The model with loaded LoRAs (typically generator.model)
        """
        lora_scales = kwargs.get("lora_scales")
        if lora_scales:
            # Use the manager that was initialized
            lora_merge_mode = getattr(self, "_lora_merge_mode", "gpu_reconstruct")

            if lora_merge_mode == "permanent_merge":
                manager = PermanentMergeLoRAManager
            elif lora_merge_mode == "runtime_peft":
                manager = PeftLoRAManager
            elif lora_merge_mode == "gpu_reconstruct":
                manager = LoRAManager
            elif lora_merge_mode == "cuda_graph_recapture":
                manager = CudaGraphRecaptureLoRAManager
            else:
                manager = LoRAManager

            self.loaded_lora_adapters = manager.update_adapter_scales(
                model=model,
                loaded_adapters=self.loaded_lora_adapters,
                scale_updates=lora_scales,
                logger_prefix=f"{self.__class__.__name__}.prepare: ",
            )
