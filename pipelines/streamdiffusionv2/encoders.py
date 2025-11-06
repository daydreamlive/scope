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

import html

import regex as re
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
from diffusers.utils import is_ftfy_available
from diffusers.utils import logging as diffusers_logging

if is_ftfy_available():
    import ftfy

logger = diffusers_logging.get_logger(__name__)


def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class StreamDiffusionV2TextEncoderStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def description(self) -> str:
        return "Text Encoder step that generates text_embeddings to guide the video generation"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return []

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("prompt"),
            InputParam("negative_prompt"),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="text embeddings used to guide the image generation",
            ),
            InputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                description="negative text embeddings used to guide the image generation",
            ),
            InputParam("attention_kwargs"),
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
            OutputParam(
                "negative_prompt_embeds",
                type_hint=torch.Tensor,
                kwargs_type="denoiser_input_fields",
                description="negative text embeddings used to guide the image generation",
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.prompt is not None and (
            not isinstance(block_state.prompt, str)
            and not isinstance(block_state.prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(block_state.prompt)}"
            )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # For streamdiffusionv2, prompt encoding is handled by prompt_blender
        # The actual prompt_embeds are set via conditional_dict in the pipeline before blocks execute
        # Skip all work if prompt_embeds already exist (which they should via conditional_dict)
        if block_state.prompt_embeds is None:
            # Only check inputs if we actually need to encode
            self.check_inputs(block_state)
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
        # If prompt_embeds already exists, skip state update to reduce overhead

        return components, state
