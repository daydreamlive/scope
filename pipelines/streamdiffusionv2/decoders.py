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


class StreamDiffusionV2PostprocessStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Postprocess step that decodes denoised latents to pixel space"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "denoised_pred",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "output",
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Decode to pixel space - direct assignment to reduce overhead
        block_state.output = components.stream.vae.stream_decode_to_pixel(block_state.denoised_pred)

        self.set_block_state(state, block_state)
        return components, state
