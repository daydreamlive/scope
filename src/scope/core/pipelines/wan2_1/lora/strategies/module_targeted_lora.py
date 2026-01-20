"""
Module-targeted LoRA manager for WAN models.

This strategy replicates LongLive's original LoRA loading behavior exactly,
providing a simple wrapper around the original LongLive LoRA functions.
"""

import logging
from typing import Any

import peft
import torch

logger = logging.getLogger(__name__)

__all__ = ["ModuleTargetedLoRAStrategy"]


class ModuleTargetedLoRAStrategy:
    """
    Simple wrapper around LongLive's original LoRA loading functions.

    This maintains exact compatibility with LongLive's LoRA behavior
    while integrating with the unified LoRA manager interface.
    """

    @staticmethod
    def _configure_lora_for_model(transformer, model_name, lora_config):
        """Configure LoRA for a WanDiffusionWrapper model.

        Copied from https://github.com/NVlabs/LongLive/blob/main/utils/lora_utils.py

        Args:
            transformer: The transformer model to apply LoRA to
            model_name: 'generator' or 'fake_score'
            lora_config: LoRA configuration

        Returns:
            lora_model: The LoRA-wrapped model
        """
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()

        # Define the specific modules we want to apply LoRA to
        if model_name == "generator":
            adapter_target_modules = [
                "CausalWanAttentionBlock",
                "BaseWanAttentionBlock",
            ]
        elif model_name == "fake_score":
            adapter_target_modules = ["WanAttentionBlock"]
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)

        target_linear_modules = list(target_linear_modules)

        # Create LoRA config
        adapter_type = lora_config.get("type", "lora")
        if adapter_type == "lora":
            peft_config = peft.LoraConfig(
                r=lora_config.get("rank", 16),
                lora_alpha=lora_config.get("alpha", None)
                or lora_config.get("rank", 16),
                lora_dropout=lora_config.get("dropout", 0.0),
                target_modules=target_linear_modules,
            )
        else:
            raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        return lora_model

    @staticmethod
    def _load_lora_checkpoint(model, lora_path: str):
        """Load LoRA checkpoint into PEFT model.

        Copied from https://github.com/NVlabs/LongLive/blob/main/utils/lora_utils.py

        Args:
            model: PEFT-wrapped model
            lora_path: Path to LoRA checkpoint file
        """
        lora_checkpoint = torch.load(lora_path, map_location="cpu")
        # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(model, lora_checkpoint)

    @staticmethod
    def load_adapter(
        model,
        lora_path: str,
        strength: float = 1.0,
        target_modules: list[str] | None = None,
        lora_config: dict[str, Any] | None = None,
    ) -> str:
        """
        Load LongLive LoRA using the original implementation.

        Args:
            model: PyTorch model
            lora_path: Path to LoRA file
            strength: Strength multiplier (unused for LongLive)
            target_modules: Unused (hardcoded to CausalWanAttentionBlock)
            lora_config: LoRA configuration dict

        Returns:
            The lora_path (used as identifier)
        """
        # Handle None lora_config
        if lora_config is None:
            lora_config = {}

        # Apply PEFT LoRA configuration (exactly like original LongLive code)
        model = ModuleTargetedLoRAStrategy._configure_lora_for_model(
            model, model_name="generator", lora_config=lora_config
        )

        # Load LoRA weights (exactly like original LongLive code)
        ModuleTargetedLoRAStrategy._load_lora_checkpoint(model, lora_path)

        return lora_path

    @staticmethod
    def load_adapters_from_list(
        model,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
        target_modules: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load LongLive LoRAs using the original implementation.

        Args:
            model: PyTorch model
            lora_configs: List with single LoRA config (LongLive only supports one)
            logger_prefix: Prefix for log messages
            target_modules: Unused

        Returns:
            List with single loaded adapter info
        """
        if len(lora_configs) != 1:
            raise ValueError(
                "ModuleTargetedLoRAStrategy only supports single LoRA adapter (LongLive style)"
            )

        config = lora_configs[0]
        lora_path = config["path"]
        scale = config.get("scale", 1.0)

        # Extract LoRA config parameters
        lora_config = {k: v for k, v in config.items() if k not in ["path", "scale"]}

        try:
            # Load using original LongLive method
            adapter_path = ModuleTargetedLoRAStrategy.load_adapter(
                model=model,
                lora_path=lora_path,
                strength=scale,
                lora_config=lora_config,
            )

            logger.info(f"{logger_prefix}Loaded LongLive LoRA: {lora_path}")

            return [{"path": adapter_path, "scale": scale}]

        except Exception as e:
            logger.error(
                f"{logger_prefix}Failed to load LongLive LoRA {lora_path}: {e}"
            )
            raise

    @staticmethod
    def update_adapter_scales(
        model,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: list[dict[str, Any]],
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """
        LongLive doesn't support runtime scale updates.
        This method exists for interface compatibility but does nothing.
        """
        logger.warning(
            f"{logger_prefix}LongLive LoRA does not support runtime scale updates"
        )
        return loaded_adapters
