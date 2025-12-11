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
            InputParam(
                "vace_context",
                default=None,
                description="Cached VACE context (persists across calls)",
            ),
            InputParam(
                "vace_ref_images",
                default=None,
                description="Cached prepared reference images (persists across calls)",
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

        # Check if VACE context is already cached from a previous pipeline call
        # This avoids re-encoding reference images on subsequent calls
        if block_state.vace_context is not None:
            logger.info("PrepareVaceContextBlock: Using cached VACE context")
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

        # Encode VACE context using reference-only approach (default)
        vace_context = self._encode_vace_context_reference_only(
            vae, prepared_refs, height, width, device
        )

        block_state.vace_context = vace_context
        block_state.vace_ref_images = [prepared_refs]

        logger.info(
            f"PrepareVaceContextBlock: Encoded VACE context with shape {vace_context[0].shape}"
        )

        self.set_block_state(state, block_state)
        return components, state

    def _encode_vace_context_reference_only(
        self, vae, prepared_refs, height, width, device
    ):
        """
        Encode VACE context using only reference images (no dummy frames).

        This is the recommended approach for R2V mode as it prevents reference
        images from appearing in the output video.

        Args:
            vae: VAE model wrapper
            prepared_refs: List of prepared reference image tensors [C, 1, H, W]
            height: Target height
            width: Target width
            device: Device to use

        Returns:
            List containing encoded VACE context tensor [C, num_refs, H, W]
        """
        # Stack refs: list of [C, 1, H, W] -> [1, C, num_refs, H, W]
        prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
        # Convert to VAE's dtype (typically bfloat16)
        vae_dtype = next(vae.parameters()).dtype
        prepared_refs_stacked = prepared_refs_stacked.to(dtype=vae_dtype)
        ref_latents_out = vae.encode_to_latent(prepared_refs_stacked, use_cache=False)

        # Convert [1, num_refs, C, H, W] -> [C, num_refs, H, W] (transpose to channel-first)
        ref_latent_batch = ref_latents_out[0].permute(1, 0, 2, 3)

        # Pad to 96 channels for VACE compatibility
        # VACE was trained with 96 channels (16 base * 6 for masked video generation)
        # For R2V mode without masks, we pad with zeros
        current_channels = ref_latent_batch.shape[0]
        if current_channels < 96:
            pad_channels = 96 - current_channels
            padding = torch.zeros(
                (
                    pad_channels,
                    ref_latent_batch.shape[1],
                    ref_latent_batch.shape[2],
                    ref_latent_batch.shape[3],
                ),
                dtype=ref_latent_batch.dtype,
                device=ref_latent_batch.device,
            )
            ref_latent_batch = torch.cat([ref_latent_batch, padding], dim=0)

        # VACE context is just the reference images (no dummy frames for R2V)
        return [ref_latent_batch]

    def _encode_vace_context_with_dummy_frames(
        self, vae, prepared_refs, height, width, device
    ):
        """
        Encode VACE context using reference images concatenated with dummy frames.

        This approach concatenates reference images with dummy (zero) frames,
        which can cause reference images to appear periodically in the output video.
        Useful for experimentation purposes.

        Args:
            vae: VAE model wrapper
            prepared_refs: List of prepared reference image tensors [C, 1, H, W]
            height: Target height
            width: Target width
            device: Device to use

        Returns:
            List containing encoded VACE context tensor [C, num_refs + 1, H, W]
        """
        # Encode reference images via VAE
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

        # Encode context (R2V mode: no masks, just reference images + dummy frames)
        vace_context = vace_encode_frames(
            vae, dummy_frames, [prepared_refs], masks=None
        )

        return vace_context
