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


class TextConditioningBlock(ModularPipelineBlocks):
    model_name = "KreaRealtimeVideo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Text Conditioning block that generates text embeddings to guide the video generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="text embeddings used to guide the image generation",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="text embeddings used to guide the image generation",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # For krea_realtime_video, prompt encoding is handled by prompt_blender
        # The actual prompt_embeds are set via conditional_dict in the pipeline before blocks execute
        # Skip all work if prompt_embeds already exist (which they should via conditional_dict)
        if block_state.prompt_embeds is None:
            # Use the text encoder if needed
            if hasattr(block_state, "prompt") and block_state.prompt is not None:
                conditional_dict = components.text_encoder(
                    text_prompts=[block_state.prompt]
                )
                block_state.prompt_embeds = conditional_dict["prompt_embeds"]
            else:
                # Default empty prompt
                conditional_dict = components.text_encoder(text_prompts=[""])
                block_state.prompt_embeds = conditional_dict["prompt_embeds"]
            self.set_block_state(state, block_state)

        return components, state
