# This is a custom version of the wan2_1 PrepareNext block that is specifically used to improve V2V behavior with RewardForcing
# See https://github.com/daydreamlive/scope/pull/229 for more context
# See inline comments below for the specific change made to the wan2_1 PrepareNext behavior
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


class PrepareNextBlock(ModularPipelineBlocks):
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
        return "Prepare Next block updates state for the next latent block after the current latent block is complete"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                default=None,
                description="Input video (if present, indicates video input is enabled)",
            ),
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

        _, num_frames, _, _, _ = block_state.latents.shape

        # This specific line is the only change made to the wan2_1 PrepareNext block
        # This is a special case for the first block that seems to help V2V with RewardForcing
        # We do not understand why this is necessary yet!
        if block_state.video is not None and block_state.current_start_frame == 0:
            num_frames -= 1

        block_state.current_start_frame += num_frames

        self.set_block_state(state, block_state)
        return components, state
