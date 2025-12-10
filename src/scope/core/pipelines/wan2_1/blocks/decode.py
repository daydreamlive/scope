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
from diffusers.utils import logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)


class DecodeBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Decode block that decodes denoised latents to pixel space"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "output_video",
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        _, num_latent_frames, _, _, _ = block_state.latents.shape

        # Decode to pixel space
        video = components.vae.decode_to_pixel(block_state.latents, use_cache=True)

        _, num_pixel_frames, _, _, _ = video.shape

        # Log VAE expansion ratio
        logger.info(
            f"DecodeBlock: VAE decoded {num_latent_frames} latent frames -> {num_pixel_frames} pixel frames "
            f"(expansion ratio: {num_pixel_frames/num_latent_frames:.2f}x)"
        )

        block_state.output_video = video

        self.set_block_state(state, block_state)
        return components, state
