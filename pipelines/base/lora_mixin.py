"""Mixin class for pipelines that support LoRA adapters."""

from pipelines.base.wan2_1.lora import LoRAManager


class LoRAEnabledPipeline:
    """
    Mixin for pipelines that support LoRA adapters.

    Provides common functionality for loading and updating LoRA adapters
    to avoid code duplication across WAN-based pipelines.
    """

    def _init_loras(self, config: dict, model) -> None:
        """
        Initialize LoRA adapters during pipeline construction.

        Args:
            config: Pipeline configuration dict that may contain 'loras' key
            model: The model to apply LoRAs to (typically generator.model)
        """
        lora_configs = config.get("loras", [])
        print(f"_init_loras: Found {len(lora_configs)} LoRA configs to load")
        self.loaded_lora_adapters = LoRAManager.load_adapters_from_list(
            model=model,
            lora_configs=lora_configs,
            logger_prefix=f"{self.__class__.__name__}.__init__: ",
        )
        print(
            f"_init_loras: Completed, loaded {len(self.loaded_lora_adapters)} LoRA adapters"
        )

    def _handle_lora_scale_updates(self, kwargs: dict, model) -> None:
        """
        Handle runtime LoRA scale updates in prepare().

        Args:
            kwargs: Keyword arguments from prepare() that may contain 'lora_scales'
            model: The model with loaded LoRAs (typically generator.model)
        """
        lora_scales = kwargs.get("lora_scales")
        if lora_scales:
            self.loaded_lora_adapters = LoRAManager.update_adapter_scales(
                model=model,
                loaded_adapters=self.loaded_lora_adapters,
                scale_updates=lora_scales,
                logger_prefix=f"{self.__class__.__name__}.prepare: ",
            )
