import logging
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    InputParam,
    OutputParam,
)

logger = logging.getLogger(__name__)


class PreparePruneMaskBlock(ModularPipelineBlocks):
    """Prepare a spatial prune mask from an external inpainting mask.

    Accepts an inpainting_mask at pixel resolution [1, 1, H, W], downsamples by 16x
    (VAE 8x, patch 2x) using avg_pool2d + threshold at 0.5, and outputs
    prune_mask: [H'*W'] boolean.

    Skips on first chunk (current_start_frame == 0) since there's no cache to
    approximate against.
    """

    @property
    def description(self) -> str:
        return "Prepare spatial prune mask from external inpainting mask"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "inpainting_mask",
                type_hint=torch.Tensor | None,
                default=None,
                description="External inpainting mask at pixel resolution [1, 1, H, W]. "
                "1.0 = regions to regenerate (keep), 0.0 = regions to prune (unchanged).",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index",
            ),
            InputParam(
                "masked_pruning_enabled",
                type_hint=bool,
                default=False,
                description="Whether masked pruning is enabled",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prune_mask",
                type_hint=torch.Tensor | None,
                description="Spatial prune mask [H'*W'] boolean, True=keep, or None",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        masked_pruning_enabled = block_state.masked_pruning_enabled
        if masked_pruning_enabled is None:
            masked_pruning_enabled = False

        inpainting_mask = block_state.inpainting_mask

        if not masked_pruning_enabled or inpainting_mask is None:
            block_state.prune_mask = None
            self.set_block_state(state, block_state)
            return components, state

        # Skip on first chunk: no cache to approximate against
        if block_state.current_start_frame == 0:
            block_state.prune_mask = None
            self.set_block_state(state, block_state)
            return components, state

        # Downsample by 16x: VAE spatial downsample (8x) * patch embedding (2x)
        # inpainting_mask: [1, 1, H, W] -> [1, 1, H/16, W/16]
        downsampled = F.avg_pool2d(inpainting_mask.float(), kernel_size=16, stride=16)
        # Threshold: >0.5 means majority of pixels in the patch are marked
        spatial_mask = (downsampled > 0.5).squeeze(0).squeeze(0)  # [H', W']

        # If nothing is marked for regeneration, skip pruning entirely
        if not spatial_mask.any():
            block_state.prune_mask = None
            self.set_block_state(state, block_state)
            return components, state

        prune_mask = spatial_mask.flatten()  # [H'*W']

        kept = prune_mask.sum().item()
        total = prune_mask.shape[0]
        logger.info(
            f"Prune mask: keeping {kept}/{total} spatial positions ({kept / total:.1%})"
        )

        block_state.prune_mask = prune_mask
        self.set_block_state(state, block_state)
        return components, state
