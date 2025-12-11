"""
Block for preparing VACE context (reference images + masks).
"""

import logging
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from ..vace_utils import (
    load_and_prepare_reference_images,
    vace_encode_frames,
)

logger = logging.getLogger(__name__)


class PrepareVaceContextBlock(ModularPipelineBlocks):
    """Prepare VACE context from reference images for conditioning."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Prepare VACE context from reference images for conditioning"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "ref_images",
                default=None,
                description="List of paths to reference images",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "vace_context",
                description="Encoded VACE context",
            ),
            OutputParam(
                "vace_ref_images",
                description="Prepared reference images",
            ),
        ]

    def __call__(
        self,
        components,
        state: PipelineState,
    ) -> tuple[Any, PipelineState]:
        """
        Prepare VACE context from reference images.

        Args:
            ref_images: List of reference image paths or None

        Returns:
            Updated state with vace_context and vace_ref_images
        """
        block_state = self.get_block_state(state)

        ref_images = block_state.ref_images

        # Skip if no reference images provided
        if ref_images is None or len(ref_images) == 0:
            block_state.vace_context = None
            block_state.vace_ref_images = None
            self.set_block_state(state, block_state)
            return components, state

        logger.info(
            f"PrepareVaceContextBlock: Preparing {len(ref_images)} reference images"
        )

        height = block_state.height
        width = block_state.width
        vae = components.vae
        device = components.config.device

        # Load and prepare reference images
        prepared_refs = load_and_prepare_reference_images(
            ref_images, height, width, device
        )

        # Encode reference images via VAE
        # For R2V, we encode references alone (no video frames yet)
        # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
        prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
        # Convert to VAE's dtype (typically bfloat16)
        vae_dtype = next(vae.parameters()).dtype
        prepared_refs_stacked = prepared_refs_stacked.to(dtype=vae_dtype)
        ref_latents = vae.encode_to_latent(prepared_refs_stacked, use_cache=False)

        # Create dummy frames in pixel space (not latent space)
        # Shape should match prepared_refs: [C, F, H, W] where C=3 (RGB), F=1 (single frame)
        dummy_frames = [
            torch.zeros((3, 1, height, width), device=device, dtype=vae_dtype)
        ]

        # Encode context (R2V mode: no masks, just reference images)
        vace_context = vace_encode_frames(
            vae, dummy_frames, [prepared_refs], masks=None
        )

        block_state.vace_context = vace_context
        block_state.vace_ref_images = [prepared_refs]

        logger.info(
            f"PrepareVaceContextBlock: Encoded VACE context with shape {vace_context[0].shape}"
        )

        self.set_block_state(state, block_state)
        return components, state
