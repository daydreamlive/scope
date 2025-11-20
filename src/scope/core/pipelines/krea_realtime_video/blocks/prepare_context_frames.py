import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)


class PrepareContextFramesBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Prepare Context Frames block updates buffers tracking context frames used in KV cache recomputation"

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
                "output_video",
                required=True,
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
            ),
            InputParam(
                "context_frame_buffer_max_size",
                type_hint=int,
                description="Max size of context frame buffer",
            ),
            InputParam(
                "decoded_frame_buffer_max_size",
                type_hint=int,
                description="Max size of decoded frame buffer",
            ),
            InputParam(
                "context_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of latent frames",
            ),
            InputParam(
                "decoded_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of decoded frames",
            ),
            InputParam(
                "first_context_frame",
                type_hint=torch.Tensor,
                description="First frame of first block",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "first_context_frame",
                type_hint=torch.Tensor,
                description="First frame of first block",
            ),
            OutputParam(
                "context_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of latent frames",
            ),
            OutputParam(
                "decoded_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of decoded frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[any, PipelineState]:
        block_state = self.get_block_state(state)

        if block_state.current_start_frame == 0:
            block_state.first_context_frame = block_state.latents[:, :1]

        if block_state.context_frame_buffer_max_size > 0:
            block_state.context_frame_buffer = torch.cat(
                [
                    block_state.context_frame_buffer,
                    block_state.latents.to(
                        block_state.context_frame_buffer.device,
                        block_state.context_frame_buffer.dtype,
                    ),
                ],
                dim=1,
            )[:, -block_state.context_frame_buffer_max_size :]

        if block_state.decoded_frame_buffer_max_size > 0:
            block_state.decoded_frame_buffer = torch.cat(
                [
                    block_state.decoded_frame_buffer,
                    block_state.output_video.to(
                        block_state.decoded_frame_buffer.device,
                        block_state.decoded_frame_buffer.dtype,
                    ),
                ],
                dim=1,
            )[:, -block_state.decoded_frame_buffer_max_size :]

        self.set_block_state(state, block_state)
        return components, state
