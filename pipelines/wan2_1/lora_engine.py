"""LoRA engine for WAN-based models.

This module centralizes LoRA loading and scale updates so that modular
pipelines can stay thin and block graphs remain agnostic to LoRA details.

Initial implementation focuses on two strategies:
- ``runtime_peft``: LoRA via PEFT with runtime scale updates.
- ``permanent_merge``: LoRA merged into base weights at load time,
  with no runtime scale updates.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import peft
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoadedAdapter:
    path: str
    scale: float
    adapter_name: str


class LoRAEngine:
    """Facade for loading and updating LoRA adapters on WAN models."""

    @staticmethod
    def _sanitize_adapter_name(path: str) -> str:
        """Generate a stable adapter name from a filesystem path."""
        name = Path(path).stem
        # Replace characters that are problematic in module names
        for ch in (".", "/", "\\", " ", "-"):
            name = name.replace(ch, "_")
        return name

    @staticmethod
    def _find_target_linear_modules(model: nn.Module) -> list[str]:
        """Find Linear modules inside WAN attention blocks.

        We mirror the LongLive LoRA targeting logic and look for all Linear
        submodules inside modules whose class is ``CausalWanAttentionBlock``
        or ``WanAttentionBlock``. This works across the WAN variants used in
        the modular pipelines.
        """
        target_linear_modules: set[str] = set()
        adapter_target_modules = {"CausalWanAttentionBlock", "WanAttentionBlock"}

        for name, module in model.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, nn.Linear):
                        target_linear_modules.add(full_name)

        result = sorted(target_linear_modules)
        if not result:
            logger.warning(
                "LoRAEngine._find_target_linear_modules: "
                "no Linear modules found inside WAN attention blocks; "
                "LoRA will have no effect."
            )
        return result

    # --------------------------------------------------------------------- #
    # Runtime PEFT strategy helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_lora_config(
        target_modules: list[str],
        rank: int = 16,
        alpha: int | None = None,
        dropout: float = 0.0,
    ) -> peft.LoraConfig:
        """Construct a PEFT LoRAConfig for WAN models."""
        if alpha is None:
            alpha = rank

        return peft.LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=False,
        )

    @staticmethod
    def _load_peft_adapter(
        model: nn.Module,
        lora_path: str,
        adapter_name: str,
        logger_prefix: str,
    ) -> peft.PeftModel:
        """Wrap model with PEFT and load a single LoRA adapter from path.

        Notes:
            - For now we support a single adapter per model instance.
            - LoRA checkpoints are expected to be PEFT-compatible state dicts.
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"{logger_prefix}LoRA file not found: {lora_path}")

        target_linear_modules = LoRAEngine._find_target_linear_modules(model)
        if not target_linear_modules:
            logger.warning(
                "%sNo target Linear modules found; PEFT LoRA will not modify the model.",
                logger_prefix,
            )

        lora_config = LoRAEngine._build_lora_config(
            target_modules=target_linear_modules
        )

        # Wrap base model with PEFT if it is not already wrapped
        if not isinstance(model, peft.PeftModel):
            peft_model = peft.get_peft_model(model, lora_config)
        else:
            # If already PEFT-wrapped, reuse it (single adapter scenario)
            peft_model = model

        # Load LoRA weights
        state_dict = torch.load(lora_path, map_location="cpu")
        try:
            peft.set_peft_model_state_dict(peft_model, state_dict)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"{logger_prefix}Failed to load PEFT LoRA weights from {lora_path}: {exc}"
            ) from exc

        logger.info(
            "%sLoaded runtime_peft LoRA adapter from %s into model",
            logger_prefix,
            lora_path,
        )
        return peft_model

    @staticmethod
    def _update_peft_scales(
        model: peft.PeftModel,
        adapter_name: str,
        new_scale: float,
        logger_prefix: str,
    ) -> None:
        """Update LoRA scaling factor for a PEFT adapter."""
        from peft.tuners.lora import LoraLayer

        updated_layers = 0
        for _, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if adapter_name in module.scaling:
                    module.scaling[adapter_name] = float(new_scale)
                    updated_layers += 1

        logger.info(
            "%sUpdated runtime_peft LoRA scale for adapter '%s' to %.3f on %d layers",
            logger_prefix,
            adapter_name,
            new_scale,
            updated_layers,
        )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    @staticmethod
    def load_adapters_from_list(
        model: nn.Module,
        lora_configs: list[dict[str, Any]],
        merge_mode: str | None,
        logger_prefix: str = "",
    ) -> tuple[nn.Module, list[dict[str, Any]]]:
        """Load LoRA adapters according to the requested merge mode.

        For now we support at most one LoRA file per model instance. If more
        than one is specified, only the first is used and a warning is logged.
        """
        if not lora_configs:
            return model, []

        merge_mode = merge_mode or "runtime_peft"

        # Take only the first config for now to keep implementation simple.
        primary = lora_configs[0]
        if len(lora_configs) > 1:
            logger.warning(
                "%sMultiple LoRAs requested (%d). Only the first will be applied for now.",
                logger_prefix,
                len(lora_configs),
            )

        path = primary.get("path")
        scale = float(primary.get("scale", 1.0))
        if not path:
            raise ValueError(f"{logger_prefix}LoRA config is missing 'path'")

        adapter_name = LoRAEngine._sanitize_adapter_name(path)
        loaded: LoadedAdapter = LoadedAdapter(
            path=path, scale=scale, adapter_name=adapter_name
        )

        if merge_mode == "permanent_merge":
            # Load via PEFT and immediately merge weights into the base model.
            peft_model = LoRAEngine._load_peft_adapter(
                model=model,
                lora_path=path,
                adapter_name=adapter_name,
                logger_prefix=logger_prefix,
            )
            merged_model = peft_model.merge_and_unload()
            logger.info(
                "%sApplied permanent_merge LoRA from %s (scale %.3f)",
                logger_prefix,
                path,
                scale,
            )
            return merged_model, [loaded.__dict__]

        if merge_mode == "runtime_peft":
            peft_model = LoRAEngine._load_peft_adapter(
                model=model,
                lora_path=path,
                adapter_name=adapter_name,
                logger_prefix=logger_prefix,
            )
            # Apply initial scale
            LoRAEngine._update_peft_scales(
                model=peft_model,
                adapter_name=adapter_name,
                new_scale=scale,
                logger_prefix=logger_prefix,
            )
            return peft_model, [loaded.__dict__]

        # Unknown merge mode – do not modify the model, but log.
        logger.warning(
            "%sUnknown lora_merge_mode '%s'; skipping LoRA application.",
            logger_prefix,
            merge_mode,
        )
        return model, [loaded.__dict__]

    @staticmethod
    def update_adapter_scales(
        model: nn.Module,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: Iterable[dict[str, Any]],
        merge_mode: str | None,
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Update scales for already-loaded adapters.

        For ``permanent_merge`` this is a no-op. For ``runtime_peft`` the
        scales are applied to the PEFT model in-place.
        """
        merge_mode = merge_mode or "runtime_peft"
        if not loaded_adapters:
            return loaded_adapters

        if merge_mode == "permanent_merge":
            # Scales are baked into weights; nothing to do.
            logger.info(
                "%sIgnoring lora_scales update for permanent_merge strategy",
                logger_prefix,
            )
            return loaded_adapters

        if merge_mode != "runtime_peft":
            logger.warning(
                "%supdate_adapter_scales: unsupported merge_mode '%s'; no-op",
                logger_prefix,
                merge_mode,
            )
            return loaded_adapters

        if not isinstance(model, peft.PeftModel):
            logger.warning(
                "%supdate_adapter_scales: model is not a PeftModel; no-op",
                logger_prefix,
            )
            return loaded_adapters

        # We only support a single adapter for now.
        adapter_info = loaded_adapters[0]
        adapter_path = adapter_info.get("path")
        adapter_name = adapter_info.get(
            "adapter_name"
        ) or LoRAEngine._sanitize_adapter_name(adapter_path)

        # Find matching update (by path)
        new_scale = None
        for update in scale_updates:
            if update.get("path") == adapter_path:
                new_scale = float(update.get("scale", adapter_info.get("scale", 1.0)))
                break

        if new_scale is None:
            return loaded_adapters

        LoRAEngine._update_peft_scales(
            model=model,
            adapter_name=adapter_name,
            new_scale=new_scale,
            logger_prefix=logger_prefix,
        )

        adapter_info["scale"] = new_scale
        return [adapter_info]
