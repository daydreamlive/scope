from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    InputParam,
    OutputParam,
)

from scope.core.pipelines.process import preprocess_chunk


class PreprocessVideoBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def description(self) -> str:
        return "Preprocess Video block transforms video so that it is ready for downstream blocks"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                default=None,
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                description="Input video to convert into noisy latents",
            ),
            InputParam(
                "vace_input_frames",
                default=None,
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                description="Input frames for VACE conditioning",
            ),
            InputParam(
                "height",
                required=True,
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                required=True,
                type_hint=int,
                description="Width of the video",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video",
                type_hint=torch.Tensor,
                description="Input video to convert into noisy latents",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        if block_state.video is not None and isinstance(block_state.video, list):
            block_state.video = preprocess_chunk(
                block_state.video,
                components.config.device,
                components.config.dtype,
                height=block_state.height,
                width=block_state.width,
            )

        if block_state.vace_input_frames is not None and isinstance(
            block_state.vace_input_frames, list
        ):
            block_state.vace_input_frames = preprocess_chunk(
                block_state.vace_input_frames,
                components.config.device,
                components.config.dtype,
                height=block_state.height,
                width=block_state.width,
            )

        self.set_block_state(state, block_state)
        return components, state
