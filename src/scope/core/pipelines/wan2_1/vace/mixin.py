"""Mixin for pipelines that support VACE (Video All-In-One Creation and Editing).

This mixin handles upfront VACE loading at pipeline initialization, treating VACE
as a toggle similar to permanent merge LoRA strategy. Pipelines using this mixin
can enable VACE support by loading weights at initialization time rather than lazily.

VACE Composition Strategy:
When LoRA is present, VACE must be applied in the correct order to avoid model degradation:
- LongLive: VACE wraps base model, then LoRA wraps VACE (PeftModel(VACE(CausalWan)))
- StreamDiffusionV2: VACE wraps base model, then LoRA wraps VACE (PeftModel(VACE(CausalWan)))

This ensures LoRA is the outermost wrapper, which is critical for maintaining generation quality.

Memory Requirements:
VACE with the Mixin approach requires ~48GB VRAM minimum. The text encoder (~6.4GB in FP8)
remains on GPU for optimal prompt encoding performance.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Get value from config whether it's dict-like or object-like."""
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


class VACEEnabledPipeline:
    """Shared VACE integration for Wan2.1-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_vace(config, model, device, dtype, quantization, text_encoder)` during __init__
      to wrap the model with VACE support if vace_path is provided in config.

    The mixin handles:
    - Wrapping the base model with CausalVaceWanModel
    - Loading VACE-specific weights from checkpoint
    - Moving VACE components to the correct device/dtype
    - Optional FP8 quantization of VACE components
    - PEFT compatibility (VACE must be loaded before LoRA for correct ordering)

    The mixin keeps track of:
    - self.vace_enabled: Whether VACE is active
    - self.vace_path: Path to VACE checkpoint (if any)
    - self.vace_in_dim: Input dimension for VACE (default 96)
    - self.vace_layers: List of layer indices where VACE is applied (default every 2nd)

    Architecture Compatibility:
    This implementation is specifically designed for Wan2.1 architecture models
    and works with any CausalWanModel implementation (LongLive, StreamDiffusionV2,
    KreaRealtimeVideo, RewardForcing, etc.).
    """

    vace_enabled: bool = False
    vace_path: str | None = None
    vace_in_dim: int = 96
    vace_layers: list[int] | None = None

    def _init_vace(
        self,
        config: Any,
        model: Any,
        device,
        dtype,
        quantization=None,
    ) -> Any:
        """Initialize VACE support if vace_path is provided in config.

        Args:
            config: Pipeline configuration that may contain:
                - 'vace_path': Path to VACE checkpoint (enables VACE)
                - 'vace_in_dim': Input dimension for VACE (default 96)
                - 'vace_layers': List of layer indices where VACE is applied (default every 2nd)
            model: Underlying diffusion model (CausalWanModel or wrapped version)
            device: Target device for VACE components
            dtype: Target dtype for VACE components
            quantization: Optional quantization config (e.g., Quantization.FP8_E4M3FN)

        Returns:
            Model instance, possibly wrapped with CausalVaceWanModel if VACE is enabled.
        """
        from ..vace import CausalVaceWanModel, load_vace_weights_only

        vace_path = _get_config_value(config, "vace_path")
        vace_in_dim = _get_config_value(config, "vace_in_dim", 96)
        vace_layers = _get_config_value(config, "vace_layers", None)

        # Get vace_in_dim and vace_layers from base_model_kwargs if present
        base_model_kwargs = _get_config_value(config, "base_model_kwargs")
        if base_model_kwargs:
            if "vace_in_dim" in base_model_kwargs:
                vace_in_dim = base_model_kwargs["vace_in_dim"]
            if "vace_layers" in base_model_kwargs:
                vace_layers = base_model_kwargs["vace_layers"]

        # Try to get from model_config if available
        model_config = _get_config_value(config, "model_config")
        if model_config:
            if hasattr(model_config, "vace_in_dim"):
                vace_in_dim = model_config.vace_in_dim
            if hasattr(model_config, "vace_layers"):
                vace_layers = model_config.vace_layers

        self.vace_path = vace_path
        self.vace_in_dim = vace_in_dim
        self.vace_layers = vace_layers
        self.vace_enabled = False

        if not vace_path:
            logger.info("_init_vace: No vace_path provided, VACE disabled")
            return model

        logger.debug(
            f"_init_vace: Loading VACE support upfront "
            f"(vace_in_dim={vace_in_dim}, vace_layers={vace_layers})"
        )

        # Wrap model with VACE
        start = time.time()
        vace_wrapped_model = CausalVaceWanModel(
            model, vace_in_dim=vace_in_dim, vace_layers=vace_layers
        )
        logger.info(
            f"_init_vace: Wrapped model with VACE in {time.time() - start:.3f}s"
        )

        # Move VACE-specific components to correct device/dtype
        # The wrapped model's VACE components (vace_patch_embedding, vace_blocks) were created
        # on CPU with default dtype. We need to move them to match the base model.
        vace_wrapped_model.vace_patch_embedding.to(device=device, dtype=dtype)
        vace_wrapped_model.vace_blocks.to(device=device, dtype=dtype)

        # Load VACE weights
        start = time.time()
        load_vace_weights_only(vace_wrapped_model, vace_path)
        logger.info(f"_init_vace: Loaded VACE weights in {time.time() - start:.3f}s")

        # Quantize VACE components if quantization is enabled
        if quantization is not None:
            # Import here to avoid circular dependency
            try:
                from ...utils import Quantization

                if quantization == Quantization.FP8_E4M3FN:
                    logger.info(
                        "_init_vace: Quantizing VACE components to FP8 (matching base model)..."
                    )
                    start = time.time()

                    from torchao.quantization.quant_api import (
                        Float8DynamicActivationFloat8WeightConfig,
                        PerTensor,
                        quantize_,
                    )

                    quantize_(
                        vace_wrapped_model.vace_patch_embedding,
                        Float8DynamicActivationFloat8WeightConfig(
                            granularity=PerTensor()
                        ),
                        device=device,
                    )
                    quantize_(
                        vace_wrapped_model.vace_blocks,
                        Float8DynamicActivationFloat8WeightConfig(
                            granularity=PerTensor()
                        ),
                        device=device,
                    )
                    logger.info(
                        f"_init_vace: Quantized VACE to FP8 in {time.time() - start:.3f}s"
                    )
            except ImportError:
                logger.warning(
                    "_init_vace: Could not import Quantization, skipping quantization check"
                )

        self.vace_enabled = True
        logger.info("_init_vace: VACE enabled successfully")

        return vace_wrapped_model
