"""Mixin for pipelines that support masked pruning (token pruning with external masks).

This mixin handles wrapping the model with CausalPrunedWanModel at pipeline
initialization. No weight loading is needed since the wrapper reuses existing weights.

Composition order: LoRA -> MaskedPruning -> VACE -> CausalWanModel (base)
MaskedPruning must be applied after VACE but before LoRA.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class MaskedPruningEnabledPipeline:
    """Shared masked pruning integration for Wan2.1-based pipelines.

    Pipelines using this mixin are expected to:
    - Call `_init_masked_pruning(config, model)` during __init__
      after VACE init, before LoRA wrapping.

    The mixin handles:
    - Wrapping the model with CausalPrunedWanModel
    - No weight loading needed (reuses existing weights)
    """

    masked_pruning_enabled: bool = False

    def _init_masked_pruning(self, config: Any, model: Any) -> Any:
        """Initialize masked pruning support if enabled in config.

        Args:
            config: Pipeline configuration that may contain 'masked_pruning_enabled'
            model: Underlying diffusion model (CausalWanModel or wrapped version)

        Returns:
            Model instance, possibly wrapped with CausalPrunedWanModel.
        """
        from .models.causal_pruned_model import CausalPrunedWanModel

        enabled = getattr(config, "masked_pruning_enabled", False)
        if hasattr(config, "get"):
            enabled = config.get("masked_pruning_enabled", enabled)

        self.masked_pruning_enabled = False

        if not enabled:
            logger.info(
                "_init_masked_pruning: masked_pruning_enabled not set, pruning disabled"
            )
            return model

        logger.debug("_init_masked_pruning: Wrapping model with CausalPrunedWanModel")

        start = time.time()
        pruned_model = CausalPrunedWanModel(model)
        logger.info(
            f"_init_masked_pruning: Wrapped model with pruning in {time.time() - start:.3f}s"
        )

        self.masked_pruning_enabled = True
        logger.info("_init_masked_pruning: Masked pruning enabled successfully")

        return pruned_model
