"""
GPU reconstruction LoRA manager for WAN models - fast inference with runtime updates.

This implementation stores original weights and LoRA diffs on GPU, reconstructing
weights on-demand when scales change. Provides excellent inference performance
(~10 FPS) but slow update times (~60s) and slow initial load time.

PERFORMANCE CHARACTERISTICS:
- Inference: 10.3 FPS (excellent, matches permanent_merge)
- Updates: ~60s (slow, due to FP8 conversion overhead)
- Init time: ~130s (one-time cost)
- Memory: Minimal overhead (stores originals in native dtype)

OPTIMIZATION ATTEMPTS (FAILED):
1. Float32 storage (WORSE):
   - Stored originals/diffs in float32 to avoid FP8 conversion
   - Result: 8.7 FPS, 99s updates (slower!)
   - Problem: Cloning 400 float32 tensors + FP32→FP8 conversion = 248ms per weight

2. Batch operations (NO IMPROVEMENT):
   - Pre-collected diffs, used torch.no_grad()
   - Result: No measurable improvement
   - Problem: Python loop forces synchronization on each weight

3. In-place operations (FAILED):
   - Tried to avoid cloning with in-place ops
   - Result: Broke FP8 quantization
   - Problem: FP8 tensors don't support in-place arithmetic (torchao limitation)

ROOT CAUSE:
- FP8 tensors require float32 conversion for arithmetic
- Each weight: FP8→float32 (convert) → add → float32→FP8 (convert)
- 400 weights × 150ms per weight = 60s
- Cannot batch due to Python loop synchronization
- Would need custom CUDA kernel to improve, or theoretically add the operators to torchao

CONCLUSION:
This approach hits a fundamental wall. The 60s update time cannot be significantly
improved without custom CUDA kernels. Better alternatives:
- permanent_merge: 9.15 FPS, no updates (production)
- runtime_peft: 4.34 FPS, <1s updates (development)

This strategy is NOT RECOMMENDED for production use.
"""
from typing import Dict, Any, List
import os
import time
import logging
from pathlib import Path
import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

__all__ = ["GpuReconstructLoRAManager"]


class GpuReconstructLoRAManager:
    """
    Manages LoRA adapters via GPU-based weight reconstruction.

    Stores original weights and LoRA diffs in float32 on GPU. When scales change,
    reconstructs weights: W_new = W_original + scale * lora_diff, then converts
    to FP8. All arithmetic is done in float32 to avoid FP8 conversion overhead.

    Trade-off: Moderate update times (~2-5s) but excellent inference performance
    due to no per-frame overhead. Uses extra VRAM to store float32 copies.

    Compatible with FP8 quantization.
    """

    # Store state per model instance (using id(model) as key)
    _model_states = {}

    @staticmethod
    def _get_model_state(model: torch.nn.Module) -> Dict[str, Any]:
        """Get or create state dict for a model instance."""
        model_id = id(model)
        if model_id not in GpuReconstructLoRAManager._model_states:
            GpuReconstructLoRAManager._model_states[model_id] = {
                "original_weights": {},  # Stores original weights in float32
                "loras": {},  # {path: {"diffs": {key: tensor}, "scale": float}}
                "backed_up": False  # Flag to track if originals are stored
            }
        return GpuReconstructLoRAManager._model_states[model_id]

    @staticmethod
    def _normalize_lora_key(lora_base_key: str) -> str:
        """
        Normalize LoRA base key to match model state dict format.

        Handles various LoRA naming conventions:
        - lora_unet_blocks_0_cross_attn_k -> blocks.0.cross_attn.k
        - diffusion_model.blocks.0.cross_attn.k -> blocks.0.cross_attn.k
        - blocks.0.cross_attn.k -> blocks.0.cross_attn.k

        Args:
            lora_base_key: Base key from LoRA file (without .lora_A/B/up/down.weight)

        Returns:
            Normalized key that matches model state dict format
        """
        if lora_base_key.startswith("lora_unet_"):
            key = lora_base_key[len("lora_unet_"):]
            import re
            key = re.sub(r'_(\d+)_', r'.\1.', key)
            key = key.replace('_', '.')
            return key

        if lora_base_key.startswith("diffusion_model."):
            return lora_base_key[len("diffusion_model."):]

        return lora_base_key

    @staticmethod
    def _build_key_map(model_state_dict: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        Build mapping from LoRA keys to model state dict keys.

        Handles multiple key formats:
        - Standard: LoRA keys like "blocks.0.attn.k" -> model "blocks.0.attn.k.weight"
        - ComfyUI: LoRA keys like "diffusion_model.blocks.0.attn.k" -> model "blocks.0.attn.k.weight"
        - PEFT-wrapped: LoRA keys like "diffusion_model.blocks.0.attn.k" -> model "base_model.model.blocks.0.attn.k.base_layer.weight"
        - Underscore format: LoRA keys like "lora_unet_blocks_0_attn_k" -> model "blocks.0.attn.k.weight"

        Args:
            model_state_dict: Model's state dict

        Returns:
            Dictionary mapping LoRA key patterns to actual model keys
        """
        key_map = {}
        is_peft_wrapped = any(k.startswith("base_model.") for k in model_state_dict.keys())

        for k in model_state_dict.keys():
            if k.endswith(".weight"):
                base_key = k[:-len(".weight")]
                key_map[base_key] = k

                if is_peft_wrapped and k.startswith("base_model.model.") and k.endswith(".base_layer.weight"):
                    peft_stripped = k[len("base_model.model."):-len(".base_layer.weight")]
                    key_map[peft_stripped] = k
                    key_map[f"diffusion_model.{peft_stripped}"] = k
                else:
                    key_map[f"diffusion_model.{base_key}"] = k

        return key_map

    @staticmethod
    def _backup_original_weights(model: torch.nn.Module, keys_to_backup: List[str]) -> None:
        """
        Backup original model weights on GPU for fast reconstruction.
        Stores originals in their native dtype (FP8) on GPU.

        NOTE: We tried storing in float32 to avoid conversion overhead, but it made
        things WORSE (99s vs 60s updates) because cloning float32 + FP32→FP8 conversion
        was slower than just doing FP8→float32→FP8.
        """
        state = GpuReconstructLoRAManager._get_model_state(model)

        if state["backed_up"]:
            return

        param_dict = dict(model.named_parameters())

        for key in keys_to_backup:
            if key in param_dict and key not in state["original_weights"]:
                # Clone and keep in original dtype (no conversion)
                state["original_weights"][key] = param_dict[key].data.clone()

        state["backed_up"] = True
        logger.debug(f"_backup_original_weights: Backed up {len(state['original_weights'])} weights on GPU")

    @staticmethod
    def _apply_all_loras(model: torch.nn.Module, initial_load: bool = False) -> None:
        """
        Reconstruct weights from GPU-stored originals + all LoRA diffs.
        Uses direct parameter assignment to preserve Float8Tensor state for FP8 quantization.

        This is the BEST PERFORMING version we found (10.3 FPS inference).
        Updates are slow (~60s) but we cannot improve without custom CUDA kernels.

        Args:
            model: PyTorch model
            initial_load: If True, this is the first application
        """
        state = GpuReconstructLoRAManager._get_model_state(model)

        if not state["backed_up"]:
            return

        timings = {}
        t_start = time.time()

        # Build new state dict from originals + all LoRA contributions
        updated_weights = {}

        t_loop_start = time.time()
        for key, original in state["original_weights"].items():
            # Start from original (already on GPU, in FP8)
            # Clone to avoid modifying the backup
            new_weight = original.clone()

            # Add all LoRA contributions
            for path, lora_info in state["loras"].items():
                if key in lora_info["diffs"]:
                    scale = lora_info["scale"]
                    diff = lora_info["diffs"][key]
                    # Scale and convert diff to match weight dtype, then add
                    # This is where the FP8 conversion overhead happens (slow but unavoidable)
                    scaled_diff = (scale * diff).to(device=new_weight.device, dtype=new_weight.dtype)
                    new_weight = new_weight + scaled_diff

            updated_weights[key] = new_weight

        timings["total_loop"] = time.time() - t_loop_start

        # Use direct parameter assignment instead of load_state_dict
        # This bypasses Float8Tensor metadata validation while preserving quantization
        # CRITICAL: This is 38x faster than load_state_dict() with FP8
        t_assign_start = time.time()
        param_dict = dict(model.named_parameters())
        for name, new_value in updated_weights.items():
            if name in param_dict:
                param_dict[name].data = new_value
        timings["assignment"] = time.time() - t_assign_start

        timings["total"] = time.time() - t_start

        if not initial_load:
            print(
                f"_apply_all_loras timing: total={timings['total']:.3f}s, "
                f"loop={timings['total_loop']:.3f}s, "
                f"assignment={timings['assignment']:.3f}s, "
                f"avg_per_weight={(timings['total_loop']/len(state['original_weights']))*1000:.1f}ms"
            )

    @staticmethod
    def load_adapter(
        model: torch.nn.Module,
        lora_path: str,
        strength: float = 1.0,
    ) -> str:
        """
        Load LoRA by computing diffs and storing them for runtime scale adjustment.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier for LoRA effect (default 1.0)

        Returns:
            The lora_path (used as identifier)

        Raises:
            FileNotFoundError: If the LoRA file does not exist
        """
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"load_adapter: LoRA file not found: {lora_path}")

        state = GpuReconstructLoRAManager._get_model_state(model)

        if lora_path.endswith('.safetensors'):
            lora_state = load_file(lora_path)
        else:
            lora_state = torch.load(lora_path, map_location='cpu')

        logger.debug(f"load_adapter: Loaded {len(lora_state)} keys from {lora_path}")

        model_state = model.state_dict()
        key_map = GpuReconstructLoRAManager._build_key_map(model_state)
        logger.debug(f"load_adapter: Built key map with {len(key_map)} entries")

        lora_diffs = {}
        keys_to_backup = []
        applied_count = 0
        skipped_no_match = 0
        skipped_no_down = 0
        skipped_no_model_key = 0

        for lora_key in lora_state.keys():
            if '.lora_up.weight' in lora_key:
                base_key = lora_key.replace('.lora_up.weight', '')
                down_key = f"{base_key}.lora_down.weight"
                alpha_key = f"{base_key}.alpha"
            elif '.lora_B.weight' in lora_key:
                base_key = lora_key.replace('.lora_B.weight', '')
                down_key = f"{base_key}.lora_A.weight"
                alpha_key = f"{base_key}.alpha"
            else:
                skipped_no_match += 1
                continue

            if down_key not in lora_state:
                skipped_no_down += 1
                if applied_count == 0:
                    logger.warning(f"load_adapter: Missing down weight for {lora_key}, expected {down_key}")
                continue

            normalized_key = GpuReconstructLoRAManager._normalize_lora_key(base_key)

            model_key = key_map.get(normalized_key)
            if model_key is None:
                model_key = key_map.get(f"diffusion_model.{normalized_key}")

            if model_key is None or model_key not in model_state:
                skipped_no_model_key += 1
                if skipped_no_model_key <= 3:
                    logger.debug(f"load_adapter: No model key found for base_key={base_key}, normalized={normalized_key}")
                continue

            alpha = lora_state.get(alpha_key)
            if alpha is not None:
                alpha = alpha.item()

            lora_up = lora_state[lora_key]
            lora_down = lora_state[down_key]
            original_weight = model_state[model_key]

            lora_up = lora_up.to(device=original_weight.device)
            lora_down = lora_down.to(device=original_weight.device)

            rank = lora_down.shape[0]
            if alpha is not None:
                scale = alpha / rank
            else:
                scale = 1.0

            lora_diff = torch.mm(
                lora_up.float().flatten(start_dim=1),
                lora_down.float().flatten(start_dim=1)
            ).reshape(original_weight.shape)

            # Store the scaled diff in original dtype (alpha is baked in, strength at runtime)
            # NOTE: We tried storing in float32 to avoid conversion overhead, but it made
            # updates SLOWER (99s vs 60s). Keep in original dtype.
            lora_diffs[model_key] = (scale * lora_diff).to(original_weight.dtype)
            keys_to_backup.append(model_key)
            applied_count += 1

        if keys_to_backup:
            GpuReconstructLoRAManager._backup_original_weights(model, keys_to_backup)

        state["loras"][str(lora_path)] = {
            "diffs": lora_diffs,
            "scale": strength,
        }

        GpuReconstructLoRAManager._apply_all_loras(model, initial_load=True)

        if applied_count > 0:
            logger.info(f"load_adapter: Applied {applied_count} LoRA weight patches")
            logger.debug(
                f"load_adapter: Skipped {skipped_no_match} no match, "
                f"{skipped_no_down} no down weight, {skipped_no_model_key} no model key"
            )
        else:
            logger.warning(f"load_adapter: No LoRA patches applied (check model compatibility)")

        return str(lora_path)

    @staticmethod
    def load_adapters_from_list(
        model: torch.nn.Module,
        lora_configs: List[Dict[str, Any]],
        logger_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Load multiple LoRA adapters by patching model weights.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages

        Returns:
            List of loaded adapter info dicts with keys: path, scale
        """
        loaded_adapters = []

        if not lora_configs:
            return loaded_adapters

        for lora_config in lora_configs:
            lora_path = lora_config.get("path")
            if not lora_path:
                logger.warning(f"{logger_prefix}Skipping LoRA config with no path specified")
                continue

            scale = lora_config.get("scale", 1.0)

            start = time.time()
            try:
                returned_path = GpuReconstructLoRAManager.load_adapter(
                    model=model,
                    lora_path=lora_path,
                    strength=scale
                )

                elapsed = time.time() - start
                logger.info(f"{logger_prefix}Loaded LoRA '{Path(lora_path).name}' (scale={scale}) in {elapsed:.3f}s")

                loaded_adapters.append({
                    "path": returned_path,
                    "scale": scale,
                })
            except FileNotFoundError as e:
                logger.error(f"{logger_prefix}LoRA file not found: {lora_path}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. File not found: {lora_path}. "
                    f"Ensure the file exists in the models/lora/ directory."
                ) from e
            except Exception as e:
                logger.error(f"{logger_prefix}Failed to load LoRA adapter: {e}")
                raise RuntimeError(
                    f"{logger_prefix}LoRA loading failed. Pipeline cannot start without all configured LoRAs. "
                    f"Error: {e}"
                ) from e

        return loaded_adapters

    @staticmethod
    def update_adapter_scales(
        model: torch.nn.Module,
        loaded_adapters: List[Dict[str, Any]],
        scale_updates: List[Dict[str, Any]],
        logger_prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Update scales for loaded LoRA adapters at runtime.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages

        Returns:
            Updated loaded_adapters list with new scale values
        """
        if not scale_updates:
            return loaded_adapters

        state = GpuReconstructLoRAManager._get_model_state(model)

        scale_map = {}
        for update in scale_updates:
            path = update.get("path")
            scale = update.get("scale")
            if path is not None and scale is not None:
                scale_map[path] = scale

        if not scale_map:
            return loaded_adapters

        scales_changed = False
        for path, new_scale in scale_map.items():
            if path in state["loras"]:
                old_scale = state["loras"][path]["scale"]
                if abs(old_scale - new_scale) > 1e-6:
                    state["loras"][path]["scale"] = new_scale
                    scales_changed = True
                    logger.info(f"{logger_prefix}Updated LoRA '{Path(path).name}' scale from {old_scale} to {new_scale}")

                    for adapter_info in loaded_adapters:
                        if adapter_info["path"] == path:
                            adapter_info["scale"] = new_scale
                            break

        if scales_changed:
            GpuReconstructLoRAManager._apply_all_loras(model)
            logger.debug(f"{logger_prefix}Reconstructed weights with updated scales")

        return loaded_adapters
