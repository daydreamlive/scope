import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class PrepareRecacheFramesBlock(ModularPipelineBlocks):
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
        return "Prepare Recache Frames block updates buffers tracking frames used in frame recaching"

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
                "recache_buffer",
                required=True,
                type_hint=torch.Tensor,
                description="Sliding window of recache frames",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "recache_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of recache frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[any, PipelineState]:
        block_state = self.get_block_state(state)

        block_state.recache_buffer = torch.cat(
            [
                block_state.recache_buffer[:, components.config.num_frame_per_block :],
                block_state.latents.to(block_state.recache_buffer.device),
            ],
            dim=1,
        )

        self.set_block_state(state, block_state)
        return components, state
