"""
LoRA utilities for WAN models - thin wrapper that delegates to strategy implementations.

This module provides a unified interface for different LoRA merge strategies:
- permanent_merge: Maximum FPS, no runtime updates (permanent_merge_lora.py)
- runtime_peft: Instant updates with per-frame overhead (peft_lora.py)
- gpu_reconstruct: Fast FPS with moderate update times (gpu_reconstruct_lora.py)
- cuda_graph_recapture: Fast FPS with fast updates via CUDA Graph optimization (cuda_graph_recapture_lora.py)

Supports local .safetensors and .bin files from models/lora/ directory.
"""

import logging
from typing import Any

from pipelines.wan2_1.lora.strategies.cuda_graph_recapture_lora import (
    CudaGraphRecaptureLoRAManager,
)
from pipelines.wan2_1.lora.strategies.gpu_reconstruct_lora import (
    GpuReconstructLoRAManager,
)
from pipelines.wan2_1.lora.strategies.peft_lora import PeftLoRAManager
from pipelines.wan2_1.lora.strategies.permanent_merge_lora import (
    PermanentMergeLoRAManager,
)

__all__ = ["LoRAManager"]

logger = logging.getLogger(__name__)


class LoRAManager:
    """
    Unified interface for LoRA management with multiple strategies.

    Delegates to the appropriate strategy implementation based on merge_mode.

    Available strategies:
    - permanent_merge: Merges LoRA weights permanently at load time
      + Maximum inference performance (zero overhead)
      - No runtime scale updates

    - runtime_peft: Uses PEFT LoraLayer for runtime application
      + Instant scale updates (<1s)
      - ~50% inference overhead per frame

    - gpu_reconstruct: Stores weights in float32 and reconstructs on update
      + Excellent inference performance (minimal overhead)
      + Runtime scale updates (~2-5s)
      - Extra VRAM usage (~800MB for 400 weights)

    - cuda_graph_recapture: PEFT with CUDA Graph optimization
      + Excellent inference performance (graph replay)
      + Fast scale updates (~1-5s, PEFT update + re-capture)
      - Requires static input shapes
    """

    # Default strategy if none specified
    DEFAULT_STRATEGY = "cuda_graph_recapture"

    @staticmethod
    def _get_manager_class(merge_mode: str = None):
        """Get the appropriate manager class based on merge mode."""
        if merge_mode is None:
            merge_mode = LoRAManager.DEFAULT_STRATEGY

        if merge_mode == "permanent_merge":
            return PermanentMergeLoRAManager
        elif merge_mode == "runtime_peft":
            return PeftLoRAManager
        elif merge_mode == "gpu_reconstruct":
            return GpuReconstructLoRAManager
        elif merge_mode == "cuda_graph_recapture":
            return CudaGraphRecaptureLoRAManager
        else:
            raise ValueError(
                f"Unknown merge_mode: {merge_mode}. "
                f"Supported modes: permanent_merge, runtime_peft, gpu_reconstruct, cuda_graph_recapture"
            )

    @staticmethod
    def load_adapter(
        model, lora_path: str, strength: float = 1.0, merge_mode: str = None
    ) -> str:
        """
        Load LoRA adapter using the specified merge strategy.

        Args:
            model: PyTorch model
            lora_path: Local path to LoRA file (.safetensors or .bin)
            strength: Initial strength multiplier for LoRA effect (default 1.0)
            merge_mode: Strategy to use (permanent_merge, runtime_peft, gpu_reconstruct, cuda_graph_recapture)

        Returns:
            The lora_path (used as identifier)
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)
        return manager_class.load_adapter(model, lora_path, strength)

    @staticmethod
    def load_adapters_from_list(
        model,
        lora_configs: list[dict[str, Any]],
        logger_prefix: str = "",
        merge_mode: str = None,
    ) -> list[dict[str, Any]]:
        """
        Load multiple LoRA adapters using the specified merge strategy.

        Args:
            model: PyTorch model
            lora_configs: List of dicts with keys: path (str, required), scale (float, optional, default=1.0)
            logger_prefix: Prefix for log messages
            merge_mode: Strategy to use (permanent_merge, runtime_peft, gpu_reconstruct, cuda_graph_recapture)

        Returns:
            List of loaded adapter info dicts with keys: path, scale
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)
        return manager_class.load_adapters_from_list(model, lora_configs, logger_prefix)

    @staticmethod
    def update_adapter_scales(
        model,
        loaded_adapters: list[dict[str, Any]],
        scale_updates: list[dict[str, Any]],
        logger_prefix: str = "",
        merge_mode: str = None,
    ) -> list[dict[str, Any]]:
        """
        Update scales for loaded LoRA adapters at runtime.

        Args:
            model: PyTorch model with loaded LoRAs
            loaded_adapters: List of currently loaded adapter info dicts
            scale_updates: List of dicts with 'path' and 'scale' keys
            logger_prefix: Prefix for log messages
            merge_mode: Strategy to use (must match the one used for loading)

        Returns:
            Updated loaded_adapters list with new scale values
        """
        manager_class = LoRAManager._get_manager_class(merge_mode)
        return manager_class.update_adapter_scales(
            model, loaded_adapters, scale_updates, logger_prefix
        )
