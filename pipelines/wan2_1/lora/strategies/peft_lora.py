"""
PEFT-based LoRA manager for WAN models with real-time scale updates.

This implementation uses PEFT's LoraLayer for runtime LoRA application
without weight merging, enabling instant scale updates (<1s) suitable for
real-time video generation with FP8 quantization.
"""

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from pipelines.wan2_1.lora.utils import (
    load_lora_weights,
    parse_lora_weights,
    sanitize_adapter_name,
)

logger = logging.getLogger(__name__)

__all__ = ["PeftLoRAManager"]


class PeftLoRAManager:
    """
    Manages LoRA adapters using PEFT for instant runtime scale updates.

    Unlike weight merging approaches, this wraps nn.Linear modules with
    PEFT's LoraLayer which applies LoRA in the forward pass. Scale updates
    are instant as they only modify a scaling variable.

    Compatible with torchao FP8 quantization via PEFT's torchao support.
    """

    # Store PEFT model wrapper per model instance
    _peft_models = {}

    @staticmethod
    def _get_peft_model(model: nn.Module) -> Any | None:
        """Get PEFT model wrapper if it exists."""
        model_id = id(model)
        return PeftLoRAManager._peft_models.get(model_id)

    @staticmethod
    def _set_peft_model(model: nn.Module, peft_model: Any) -> None:
        """Store PEFT model wrapper."""
        model_id = id(model)
        PeftLoRAManager._peft_models[model_id] = peft_model

    @staticmethod
    def _inject_lora_layers(
        model: nn.Module,
        lora_mapping: dict[str, dict[str, Any]],
        adapter_name: str,
        strength: float = 1.0,
    ) -> None:
        """
        Inject PEFT LoRA layers into the model.

        This wraps targeted nn.Linear modules with PEFT's LoraLayer,
        which applies LoRA in the forward pass without weight merging.
        """
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.tuners.lora import LoraLayer

        # Check if model is already a PEFT model (not just in cache)
        is_already_peft = isinstance(model, PeftModel) or hasattr(model, "base_model")

        # Determine target modules from lora_mapping
        # Use exact module paths (PEFT supports both regex and exact names)
        # Note: lora_mapping keys are normalized (from base model's state dict if PEFT-wrapped)
        target_modules = []
        for param_name in lora_mapping.keys():
            # param_name is like "blocks.0.self_attn.q.weight" (normalized)
            if param_name.endswith(".weight"):
                module_path = param_name[: -len(".weight")]

                # Verify this is actually a Linear layer in the model
                # If model is PEFT-wrapped, navigate through base_model.model
                parts = module_path.split(".")
                try:
                    if is_already_peft and hasattr(model, "base_model"):
                        # For PEFT models, navigate from base_model.model
                        current = model.base_model.model
                    else:
                        current = model
                    for part in parts:
                        current = getattr(current, part)

                    # Check if it's a Linear layer or already a LoraLayer (for already-PEFT-wrapped models)
                    # LoraLayer wraps Linear layers, so both are valid targets
                    if isinstance(current, nn.Linear):
                        target_modules.append(module_path)
                    elif isinstance(current, LoraLayer):
                        # Module is already wrapped as LoraLayer (e.g., from configure_lora_for_model)
                        # We can still add a new adapter to it
                        target_modules.append(module_path)
                        logger.debug(
                            f"_inject_lora_layers: Found LoraLayer at {module_path} (already PEFT-wrapped)"
                        )
                except AttributeError:
                    logger.debug(
                        f"_inject_lora_layers: Module {module_path} not found in model"
                    )
                    continue

        if not target_modules:
            logger.warning(
                "_inject_lora_layers: No target modules found in LoRA mapping"
            )
            return

        logger.info(
            f"_inject_lora_layers: Targeting {len(target_modules)} Linear modules"
        )
        logger.debug(f"_inject_lora_layers: Target modules: {target_modules[:5]}...")

        # Infer rank from first LoRA in mapping
        first_lora = next(iter(lora_mapping.values()))
        rank = first_lora["rank"]
        alpha = first_lora["alpha"]
        if alpha is None:
            alpha = rank  # Default alpha = rank

        # Create PEFT config with exact module paths
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            init_lora_weights=False,  # We'll load weights manually
            modules_to_save=None,  # Don't save any other modules
        )

        # Check if model already has PEFT adapters (in cache or is already PEFT)
        existing_peft_model = PeftLoRAManager._get_peft_model(model)
        if existing_peft_model is not None:
            # Add new adapter to existing PEFT model from cache
            logger.info(
                f"_inject_lora_layers: Adding adapter '{adapter_name}' to existing PEFT model"
            )
            existing_peft_model.add_adapter(adapter_name, lora_config)
            peft_model = existing_peft_model
        elif is_already_peft:
            # Model is already PEFT-wrapped but not in cache (e.g., from configure_lora_for_model)
            logger.info(
                f"_inject_lora_layers: Model is already PEFT-wrapped, adding adapter '{adapter_name}'"
            )
            # Register it in cache for future use (key by the PEFT model itself)
            PeftLoRAManager._set_peft_model(model, model)
            # Add new adapter to existing PEFT model
            model.add_adapter(adapter_name, lora_config)
            peft_model = model
        else:
            # Wrap model with PEFT
            logger.info(
                f"_inject_lora_layers: Creating new PEFT model with adapter '{adapter_name}'"
            )
            peft_model = get_peft_model(model, lora_config, adapter_name=adapter_name)
            peft_model = torch.compile(peft_model, mode="max-autotune", fullgraph=True)
            logger.info(
                "_inject_lora_layers: torch.compile applied with fullgraph=True (will error if incompatible)"
            )
            logger.info(
                f"_inject_lora_layers: Model type after compile: {type(peft_model)}"
            )
            PeftLoRAManager._set_peft_model(model, peft_model)

        # Load LoRA weights into PEFT layers
        loaded_count = 0
        for param_name, lora_info in lora_mapping.items():
            # param_name is like "blocks.0.self_attn.q.weight" (normalized)
            # PEFT wraps it with additional lora_A and lora_B parameters

            # Navigate to the PEFT-wrapped module
            module_path = (
                param_name[: -len(".weight")]
                if param_name.endswith(".weight")
                else param_name
            )

            parts = module_path.split(".")

            try:
                # Get the module (PEFT wraps model as base_model.model)
                current = peft_model.base_model.model
                for part in parts:
                    current = getattr(current, part)

                # Current should now be a LoraLayer-wrapped Linear
                if not isinstance(current, LoraLayer):
                    logger.debug(
                        f"_inject_lora_layers: {module_path} is not a LoraLayer, skipping"
                    )
                    continue

                # Load LoRA A and B weights
                lora_A_weight = lora_info["lora_A"]
                lora_B_weight = lora_info["lora_B"]

                # PEFT stores lora_A and lora_B as ModuleDict with adapter names as keys
                if adapter_name in current.lora_A:
                    current.lora_A[adapter_name].weight.data = lora_A_weight.to(
                        device=current.lora_A[adapter_name].weight.device,
                        dtype=current.lora_A[adapter_name].weight.dtype,
                    )
                    current.lora_B[adapter_name].weight.data = lora_B_weight.to(
                        device=current.lora_B[adapter_name].weight.device,
                        dtype=current.lora_B[adapter_name].weight.dtype,
                    )

                    # Set initial scaling
                    current.scaling[adapter_name] = strength

                    loaded_count += 1
                else:
                    logger.debug(
                        f"_inject_lora_layers: Adapter '{adapter_name}' not found in {module_path}"
                    )

            except AttributeError as e:
                logger.debug(
                    f"_inject_lora_layers: Could not find module {module_path}: {e}"
                )
                continue

        logger.info(f"_inject_lora_layers: Loaded {loaded_count} LoRA weight pairs")

        # Activate the adapter
        peft_model.set_adapter(adapter_name)

    @staticmethod
    def load_adapter(
        model: nn.Module,
        lora_path: str,
        strength: float = 1.0,
        adapter_name: str | None = None,
    ) -> str:
        """
        Load LoRA adapter using PEFT for runtime application.

        Args:
            model: PyTorch model
            lora_path: Path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier (default 1.0)
            adapter_name: Optional adapter name (defaults to filename)

        Returns:
            The adapter name used

        Example:
            >>> from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAManager
            >>> adapter_name = PeftLoRAManager.load_adapter(
            ...     model=pipeline.transformer,
            ...     lora_path="models/lora/my-style.safetensors",
            ...     strength=1.0
            ... )
        """
        start_time = time.time()

        if adapter_name is None:
            adapter_name = Path(lora_path).stem

        # Sanitize adapter name to ensure it's valid for PyTorch module names
        original_adapter_name = adapter_name
        adapter_name = sanitize_adapter_name(adapter_name)
        if adapter_name != original_adapter_name:
            logger.debug(
                f"load_adapter: Sanitized adapter name '{original_adapter_name}' -> '{adapter_name}'"
            )

        logger.info(
            f"load_adapter: Loading LoRA from {lora_path} as adapter '{adapter_name}'"
        )

        # Load LoRA weights
        lora_state = load_lora_weights(lora_path)
        logger.debug(f"load_adapter: Loaded {len(lora_state)} tensors from file")

        # Get model state dict (use PEFT model's state dict so build_key_map can detect PEFT wrapping)
        model_state = model.state_dict()

        # Parse and map LoRA weights to model parameters
        lora_mapping = parse_lora_weights(lora_state, model_state)

        # If model is already PEFT-wrapped, normalize the lora_mapping keys for target_modules
        # (parse_lora_weights returns PEFT paths, but target_modules needs normalized paths)
        from peft import PeftModel

        if isinstance(model, PeftModel) or hasattr(model, "base_model"):
            normalized_mapping = {}
            for key, value in lora_mapping.items():
                # Strip "base_model.model." prefix and ".base_layer" suffix
                if key.startswith("base_model.model.") and key.endswith(
                    ".base_layer.weight"
                ):
                    normalized_key = (
                        key[len("base_model.model.") : -len(".base_layer.weight")]
                        + ".weight"
                    )
                    normalized_mapping[normalized_key] = value
                else:
                    normalized_mapping[key] = value
            lora_mapping = normalized_mapping
        logger.info(
            f"load_adapter: Mapped {len(lora_mapping)} LoRA layers to model parameters"
        )

        if not lora_mapping:
            logger.warning("load_adapter: No LoRA layers matched model parameters")
            return adapter_name

        # Inject PEFT LoRA layers
        PeftLoRAManager._inject_lora_layers(model, lora_mapping, adapter_name, strength)

        elapsed = time.time() - start_time
        logger.info(f"load_adapter: Loaded adapter '{adapter_name}' in {elapsed:.3f}s")

        return adapter_name

    @staticmethod
    def load_adapters_from_list(
        model: nn.Module, lora_configs: list[dict[str, Any]], logger_prefix: str = ""
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters using PEFT.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys:
                - path (str, required)
                - scale (float, optional, default=1.0)
                - adapter_name (str, optional)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: adapter_name, path, scale

        Example:
            >>> loaded = PeftLoRAManager.load_adapters_from_list(
            ...     model=pipeline.transformer,
            ...     lora_configs=[{"path": "models/lora/style.safetensors", "scale": 1.0}]
            ... )
        """
        loaded_adapters = []

        if not lora_configs:
            return loaded_adapters

        for lora_config in lora_configs:
            lora_path = lora_config.get("path")
            if not lora_path:
                logger.warning(f"{logger_prefix}Skipping LoRA config with no path")
                continue

            scale = lora_config.get("scale", 1.0)
            adapter_name = lora_config.get("adapter_name")

            try:
                returned_adapter_name = PeftLoRAManager.load_adapter(
                    model=model,
                    lora_path=lora_path,
                    strength=scale,
                    adapter_name=adapter_name,
                )

                logger.info(
                    f"{logger_prefix}Loaded LoRA '{Path(lora_path).name}' as '{returned_adapter_name}' (scale={scale})"
                )

                loaded_adapters.append(
                    {
                        "adapter_name": returned_adapter_name,
                        "path": lora_path,
                        "scale": scale,
                    }
                )

            except FileNotFoundError as e:
                logger.error(f"{logger_prefix}LoRA file not found: {lora_path}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. File not found: {lora_path}"
                ) from e
            except Exception as e:
                logger.error(f"{logger_prefix}Failed to load LoRA: {e}", exc_info=True)
                raise RuntimeError(f"{logger_prefix}LoRA loading failed: {e}") from e

        return loaded_adapters

    @staticmethod
    def update_adapter_scales(
        model: nn.Module,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: list[dict[str, Any]],
        logger_prefix: str = "",
    ) -> list[dict[str, Any]]:
        """
        Update LoRA adapter scales at runtime (instant, <1s).

        With PEFT, scale updates only modify the scaling variable in each
        LoraLayer's forward pass, making them extremely fast.

        Args:
            model: PyTorch model with loaded LoRA adapters
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'adapter_name' (or 'path') and 'scale' keys
            logger_prefix: Prefix for log messages

        Returns:
            Updated loaded_adapters list

        Example:
            >>> self.loaded_lora_adapters = PeftLoRAManager.update_adapter_scales(
            ...     model=self.stream.generator.model,
            ...     loaded_adapters=self.loaded_lora_adapters,
            ...     scale_updates=[{"adapter_name": "my_style", "scale": 0.5}]
            ... )
        """
        if not scale_updates:
            return loaded_adapters

        peft_model = PeftLoRAManager._get_peft_model(model)
        if peft_model is None:
            logger.warning(f"{logger_prefix}No PEFT model found, cannot update scales")
            return loaded_adapters

        # Build map from adapter_name and path to scale
        scale_map = {}
        for update in scale_updates:
            adapter_name = update.get("adapter_name")
            path = update.get("path")
            scale = update.get("scale")

            if scale is None:
                continue

            if adapter_name:
                scale_map[("adapter_name", adapter_name)] = scale
            if path:
                scale_map[("path", path)] = scale

        if not scale_map:
            return loaded_adapters

        # Update scales in PEFT model
        updates_applied = 0
        for adapter_info in loaded_adapters:
            adapter_name = adapter_info.get("adapter_name")
            path = adapter_info.get("path")

            # Check if we have a scale update for this adapter
            new_scale = scale_map.get(("adapter_name", adapter_name))
            if new_scale is None:
                new_scale = scale_map.get(("path", path))

            if new_scale is None:
                continue

            old_scale = adapter_info.get("scale", 1.0)
            if abs(old_scale - new_scale) < 1e-6:
                continue

            # Update scale in all LoraLayer modules
            # Navigate through model to find all LoraLayers with this adapter
            from peft.tuners.lora import LoraLayer

            for _name, module in peft_model.named_modules():
                if isinstance(module, LoraLayer):
                    if adapter_name in module.scaling:
                        module.scaling[adapter_name] = new_scale

            # Update in loaded_adapters list
            adapter_info["scale"] = new_scale
            updates_applied += 1

            logger.info(
                f"{logger_prefix}Updated LoRA '{adapter_name}' scale: {old_scale:.3f} -> {new_scale:.3f}"
            )

        if updates_applied > 0:
            logger.debug(f"{logger_prefix}Applied {updates_applied} scale updates")

        return loaded_adapters
