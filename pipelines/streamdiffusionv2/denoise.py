# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


class StreamDiffusionV2DenoiseStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        # Note: This block depends on 'stream' rather than just 'generator' because
        # it uses the stream.inference() method, which is a high-level orchestration
        # method that manages internal state (kv_cache, crossattn_cache, conditional_dict,
        # denoising_step_list, scheduler) required for the denoising loop.
        # The stream object is a torch.nn.Module (CausalStreamInferencePipeline).
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Denoise step that performs inference using the stream pipeline"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "noisy_latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            InputParam(
                "current_start",
                required=True,
                type_hint=int,
                description="Current start position",
            ),
            InputParam(
                "current_end",
                required=True,
                type_hint=int,
                description="Current end position",
            ),
            InputParam(
                "current_step",
                required=True,
                type_hint=int,
                description="Current denoising step",
            ),
            InputParam(
                "generator",
                description="Random number generator",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "denoised_pred",
                type_hint=torch.Tensor,
                description="Denoised prediction",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Use the stream's inference method - direct call without intermediate variable
        block_state.denoised_pred = components.stream.inference(
            noise=block_state.noisy_latents,
            current_start=block_state.current_start,
            current_end=block_state.current_end,
            current_step=block_state.current_step,
            generator=block_state.generator,
        )

        self.set_block_state(state, block_state)
        return components, state
