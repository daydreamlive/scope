from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)
from diffusers.utils import logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)


class PrepareNextPixelFrameBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
        ]

    @property
    def description(self) -> str:
        return "Prepare Next block for reward_forcing that tracks pixel frames instead of latent frames for proper v2v temporal alignment"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
            ),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
            InputParam(
                "output_video",
                required=True,
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "current_start_frame",
                type_hint=int,
                description="Current starting frame index of current block",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        _, _, num_output_frames, _, _ = block_state.output_video.shape
        # Track pixel frames (not latent frames) for proper v2v temporal alignment.
        # VAE expands latents to pixels (4 latent → 13 pixel on first batch,
        # 3 latent → 12 pixel on subsequent batches), so we must track pixel space
        # to align with input video chunk extraction.
        logger.info(
            f"PrepareNextPixelFrameBlock: incrementing current_start_frame from {block_state.current_start_frame} "
            f"by {num_output_frames} pixel frames (new value: {block_state.current_start_frame + num_output_frames})"
        )
        block_state.current_start_frame += num_output_frames

        self.set_block_state(state, block_state)
        return components, state
