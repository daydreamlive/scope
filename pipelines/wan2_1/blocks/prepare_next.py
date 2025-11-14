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
        # Increment the starting index based on the # of latents actually denoised
        block_state.current_start_frame += num_frames

        self.set_block_state(state, block_state)
        return components, state
