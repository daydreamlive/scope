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
            InputParam(
                "vace_ref_images",
                default=None,
                description="VACE reference images (not used for frame stripping in this implementation)",
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

        # Decode to pixel space
        latents = block_state.latents
        video = components.vae.decode_to_pixel(latents, use_cache=True)

        # Note: VACE reference frames are NOT prepended to latents in this implementation.
        # They are only used for VACE context conditioning, so we don't need to strip them.
        # The vace_ref_images in state is used for VACE context preparation, not for frame stripping.

        block_state.output_video = video

        self.set_block_state(state, block_state)
        return components, state
