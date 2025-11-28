"""
PEFT-based LoRA manager for WAN models with real-time scale updates.

This implementation uses PEFT's LoraLayer for runtime LoRA application
without weight merging, enabling instant scale updates (<1s) suitable for
real-time video generation.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

from ..utils import (
    load_lora_weights,
    parse_lora_weights,
    sanitize_adapter_name,
    standardize_lora_for_peft,
)

logger = logging.getLogger(__name__)

__all__ = ["PeftLoRAStrategy"]

_ENABLE_TORCH_COMPILE = os.getenv("PEFT_LORA_TORCH_COMPILE", "0") == "1"


class PeftLoRAStrategy:
    """
    Manages LoRA adapters using PEFT for instant runtime scale updates.

    Unlike weight merging approaches, this wraps nn.Linear modules with
    PEFT's LoraLayer which applies LoRA in the forward pass. Scale updates
    are instant as they only modify a scaling variable.

    Compatible with torchao FP8 quantization via PEFT's torchao support.
    """

    # Store PEFT model wrapper per model instance using WeakKeyDictionary
    # to avoid memory leaks when models are garbage collected
    _peft_models: WeakKeyDictionary[nn.Module, Any] = WeakKeyDictionary()

    @staticmethod
    def _get_peft_model(model: nn.Module) -> Any | None:
        """Get PEFT model wrapper if it exists."""
        return PeftLoRAStrategy._peft_models.get(model)

    @staticmethod
    def _set_peft_model(model: nn.Module, peft_model: Any) -> None:
        """Store PEFT model wrapper."""
        PeftLoRAStrategy._peft_models[model] = peft_model

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
        # lora_mapping keys are PEFT keys if model is PEFT-wrapped (e.g., "base_model.model.blocks.0.self_attn.q.base_layer.weight")
        # We need normalized paths for target_modules (e.g., "blocks.0.self_attn.q")
        target_modules = []
        for param_name in lora_mapping.keys():
            if param_name.endswith(".weight"):
                # Normalize PEFT keys to base model paths
                if param_name.endswith(".base_layer.weight"):
                    # Handle PEFT keys with or without base_model prefix
                    if param_name.startswith("base_model.model."):
                        # PEFT key: "base_model.model.blocks.0.self_attn.q.base_layer.weight"
                        # Extract: "blocks.0.self_attn.q"
                        module_path = param_name[
                            len("base_model.model.") : -len(".base_layer.weight")
                        ]
                    else:
                        # PEFT key without prefix: "blocks.0.self_attn.q.base_layer.weight"
                        # Extract: "blocks.0.self_attn.q"
                        module_path = param_name[: -len(".base_layer.weight")]
                else:
                    # Non-PEFT key: "blocks.0.self_attn.q.weight"
                    module_path = param_name[: -len(".weight")]

                # Verify this module exists in the model
                # Navigate from model (PEFT model if wrapped) using normalized path
                parts = module_path.split(".")
                try:
                    # Always navigate from model - if PEFT-wrapped, navigate through base_model.model
                    if is_already_peft and hasattr(model, "base_model"):
                        current = model.base_model.model
                    else:
                        current = model
                    for part in parts:
                        current = getattr(current, part)

                    # Check if it's a Linear layer or already a LoraLayer
                    if isinstance(current, nn.Linear):
                        target_modules.append(module_path)
                    elif isinstance(current, LoraLayer):
                        # Module is already wrapped as LoraLayer - can add new adapter
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

        logger.debug(
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
        existing_peft_model = PeftLoRAStrategy._get_peft_model(model)
        logger.debug(
            f"_inject_lora_layers: existing_peft_model from cache: {existing_peft_model is not None}"
        )
        logger.debug(f"_inject_lora_layers: is_already_peft: {is_already_peft}")

        if existing_peft_model is not None:
            # Add new adapter to existing PEFT model from cache
            logger.debug(
                f"_inject_lora_layers: Adding adapter '{adapter_name}' to existing PEFT model"
            )
            logger.debug(
                f"_inject_lora_layers: Existing PEFT model type: {type(existing_peft_model)}"
            )
            logger.debug(
                f"_inject_lora_layers: Existing adapters: {existing_peft_model.peft_config.keys() if hasattr(existing_peft_model, 'peft_config') else 'N/A'}"
            )
            existing_peft_model.add_adapter(adapter_name, lora_config)
            peft_model = existing_peft_model
        elif is_already_peft:
            # Model is already PEFT-wrapped but not in cache (e.g., from configure_lora_for_model)
            logger.debug(
                f"_inject_lora_layers: Model is already PEFT-wrapped, adding adapter '{adapter_name}'"
            )
            # Register it in cache for future use (key by the PEFT model itself)
            PeftLoRAStrategy._set_peft_model(model, model)
            # Add new adapter to existing PEFT model
            model.add_adapter(adapter_name, lora_config)
            peft_model = model
        else:
            # Wrap model with PEFT
            logger.debug(
                f"_inject_lora_layers: Creating new PEFT model with adapter '{adapter_name}'"
            )
            peft_model = get_peft_model(model, lora_config, adapter_name=adapter_name)

            # Optionally compile for additional performance where supported.
            if _ENABLE_TORCH_COMPILE and torch.cuda.is_available():
                try:
                    peft_model = torch.compile(
                        peft_model, mode="max-autotune", fullgraph=True
                    )
                    logger.debug(
                        "_inject_lora_layers: torch.compile applied with fullgraph=True"
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning(
                        "_inject_lora_layers: torch.compile failed, running without compile: %s",
                        exc,
                    )

            logger.debug(
                f"_inject_lora_layers: Model type after PEFT wrapping: {type(peft_model)}"
            )
            PeftLoRAStrategy._set_peft_model(model, peft_model)

        # Get the underlying PEFT model (unwrap compiled wrapper if needed)
        target_model = (
            peft_model._orig_mod if hasattr(peft_model, "_orig_mod") else peft_model
        )

        # Check if we have multiple adapters - if so, we need to be careful about set_adapter
        has_multiple_adapters = (
            hasattr(target_model, "peft_config") and len(target_model.peft_config) > 1
        )

        # Only call set_adapter if this is the first adapter or if we need to initialize
        # add_adapter should initialize the structure, but set_adapter ensures it's ready
        # However, calling set_adapter will deactivate other adapters, so we avoid it for multiple adapters
        if not has_multiple_adapters:
            # For the first adapter, set it as active to initialize
            target_model.set_adapter(adapter_name)
        # For multiple adapters, add_adapter should be sufficient to initialize the structure

        # Load LoRA weights into PEFT layers
        loaded_count = 0
        missing_adapter_modules = []  # Track modules missing adapter for summary logging
        for param_name, lora_info in lora_mapping.items():
            # param_name is PEFT key if model was PEFT-wrapped (e.g., "base_model.model.blocks.0.self_attn.q.base_layer.weight")
            # or normalized key if not (e.g., "blocks.0.self_attn.q.weight")
            # After first LoRA, keys are like "blocks.0.self_attn.q.base_layer.weight" (no base_model prefix)
            # Normalize to get module path
            if param_name.endswith(".base_layer.weight"):
                # Handle both formats: with or without base_model prefix
                if param_name.startswith("base_model.model."):
                    module_path = param_name[
                        len("base_model.model.") : -len(".base_layer.weight")
                    ]
                else:
                    # Format: "blocks.0.self_attn.q.base_layer.weight" -> "blocks.0.self_attn.q"
                    module_path = param_name[: -len(".base_layer.weight")]
            elif param_name.endswith(".weight"):
                module_path = param_name[: -len(".weight")]
            else:
                module_path = param_name

            parts = module_path.split(".")

            try:
                # Navigate to the PEFT-wrapped module
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
                    # Track missing adapter for summary logging instead of logging each one
                    available_adapters = (
                        list(current.lora_A.keys())
                        if hasattr(current, "lora_A")
                        else []
                    )
                    missing_adapter_modules.append((module_path, available_adapters))

            except AttributeError as e:
                logger.debug(
                    f"_inject_lora_layers: Could not find module {module_path}: {e}"
                )
                continue

        # Log summary of missing adapters instead of individual messages
        if missing_adapter_modules:
            sample_missing = missing_adapter_modules[:3]
            available_in_sample = set()
            for _, adapters in missing_adapter_modules:
                available_in_sample.update(adapters)
            logger.warning(
                f"_inject_lora_layers: Adapter '{adapter_name}' not found in {len(missing_adapter_modules)} modules. "
                f"Sample modules: {[m[0] for m in sample_missing]}. "
                f"Available adapters in these modules: {sorted(available_in_sample)}"
            )

        logger.info(f"_inject_lora_layers: Loaded {loaded_count} LoRA weight pairs")

        # Activate all loaded adapters for multi-adapter support
        if hasattr(target_model, "peft_config"):
            all_adapters = list(target_model.peft_config.keys())
        else:
            all_adapters = [adapter_name]

        if len(all_adapters) > 1:
            # For multiple adapters, we need to activate all of them
            # PEFT's active_adapters property is read-only, so we update the private _active_adapter
            # attribute on each LoraLayer to enable simultaneous multi-adapter inference
            logger.debug(
                f"_inject_lora_layers: Activating {len(all_adapters)} adapters: {all_adapters}"
            )

            from peft.tuners.lora import LoraLayer

            updated_layers = 0
            for _name, module in target_model.named_modules():
                if isinstance(module, LoraLayer):
                    if hasattr(module, "_active_adapter"):
                        module._active_adapter = all_adapters
                        updated_layers += 1

            logger.debug(
                f"_inject_lora_layers: Activated all adapters in {updated_layers} LoraLayers"
            )
        else:
            # Only one adapter, use standard PEFT activation
            target_model.set_adapter(adapter_name)

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
            >>> from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAStrategy
            >>> adapter_name = PeftLoRAStrategy.load_adapter(
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

        # Sample LoRA keys for debugging
        sample_lora_keys = list(lora_state.keys())[:5]
        logger.debug(f"load_adapter: Sample LoRA keys from file: {sample_lora_keys}")

        # Standardize LoRA format if needed (handles lora_up/lora_down, missing diffusion_model prefix, etc.)
        converted_state = standardize_lora_for_peft(lora_path=lora_path)
        if converted_state is not None:
            logger.debug(
                f"load_adapter: Converted LoRA to PEFT format ({len(converted_state)} keys)"
            )
            lora_state = converted_state
            # Sample converted keys for debugging
            sample_converted_keys = list(lora_state.keys())[:5]
            logger.debug(
                f"load_adapter: Sample converted LoRA keys: {sample_converted_keys}"
            )

        # Always use model.state_dict() - build_key_map handles PEFT detection
        # This works for both first LoRA (non-PEFT) and subsequent LoRAs (PEFT-wrapped)
        model_state = model.state_dict()

        # Check if model is PEFT-wrapped
        from peft import PeftModel

        is_peft_wrapped = isinstance(model, PeftModel) or hasattr(model, "base_model")
        logger.debug(f"load_adapter: Model is PEFT-wrapped: {is_peft_wrapped}")
        logger.debug(f"load_adapter: Model state dict has {len(model_state)} keys")

        # Sample model state keys for debugging
        sample_model_keys = list(model_state.keys())[:5]
        logger.debug(f"load_adapter: Sample model state keys: {sample_model_keys}")

        # Parse and map LoRA weights to model parameters
        # Returns keys matching model_state (PEFT keys if model is PEFT-wrapped)
        lora_mapping = parse_lora_weights(lora_state, model_state)
        logger.info(
            f"load_adapter: Mapped {len(lora_mapping)} LoRA layers to model parameters"
        )

        # If no mapping found, log sample of what we're looking for vs what exists
        if not lora_mapping:
            logger.warning("load_adapter: No LoRA layers matched model parameters")
            # Get sample normalized LoRA keys
            from pipelines.wan2_1.lora.utils import find_lora_pair, normalize_lora_key

            sample_normalized = []
            for lora_key in list(lora_state.keys())[:5]:
                pair_result = find_lora_pair(lora_key, lora_state)
                if pair_result:
                    base_key, _, _, _ = pair_result
                    normalized = normalize_lora_key(base_key)
                    sample_normalized.append(normalized)
            logger.info(
                f"load_adapter: Sample normalized LoRA keys we're looking for: {sample_normalized}"
            )
            return adapter_name

        # Inject PEFT LoRA layers
        PeftLoRAStrategy._inject_lora_layers(
            model, lora_mapping, adapter_name, strength
        )

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
            >>> loaded = PeftLoRAStrategy.load_adapters_from_list(
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
                returned_adapter_name = PeftLoRAStrategy.load_adapter(
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
            >>> self.loaded_lora_adapters = PeftLoRAStrategy.update_adapter_scales(
            ...     model=self.stream.generator.model,
            ...     loaded_adapters=self.loaded_lora_adapters,
            ...     scale_updates=[{"adapter_name": "my_style", "scale": 0.5}]
            ... )
        """
        if not scale_updates:
            return loaded_adapters

        peft_model = PeftLoRAStrategy._get_peft_model(model)
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
