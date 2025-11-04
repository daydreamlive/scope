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
from typing import List, Optional, Union, Dict
import logging

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.configuration_utils import FrozenDict
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.utils import is_ftfy_available, logging as diffusers_logging
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)

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
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> List[ConfigSpec]:
        return []

    @property
    def inputs(self) -> List[InputParam]:
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
    def intermediate_outputs(self) -> List[OutputParam]:
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
    def __call__(
        self, components, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        # For streamdiffusionv2, prompt encoding is handled by prompt_blender
        # This block is a placeholder to maintain compatibility with Modular Diffusers
        # The actual prompt_embeds are set via conditional_dict in the pipeline
        # We just need to ensure prompt_embeds is in the state
        if block_state.prompt_embeds is None:
            # Use the stream's text encoder if needed
            if hasattr(block_state, 'prompt') and block_state.prompt is not None:
                conditional_dict = components.stream.text_encoder(text_prompts=[block_state.prompt])
                block_state.prompt_embeds = conditional_dict["prompt_embeds"]
            else:
                # Default empty prompt
                conditional_dict = components.stream.text_encoder(text_prompts=[""])
                block_state.prompt_embeds = conditional_dict["prompt_embeds"]

        # Add outputs
        self.set_block_state(state, block_state)
        return components, state


class StreamDiffusionV2PreprocessStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Preprocess step that encodes input frames to latents and adds noise"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "input",
                required=True,
                type_hint=torch.Tensor,
                description="Input video frames in BCTHW format",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Scale of noise to add to latents",
            ),
            InputParam(
                "generator",
                description="Random number generator for noise",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                default=42,
                description="Base seed for random number generation",
            ),
            InputParam(
                "current_start",
                type_hint=int,
                default=0,
                description="Current start position for seed offset",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Encoded and noised latents",
            ),
            OutputParam(
                "noisy_latents",
                type_hint=torch.Tensor,
                description="Noisy latents ready for denoising",
            ),
            OutputParam(
                "current_step",
                type_hint=int,
                description="Current denoising step",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)

        # Encode frames to latents using VAE
        latents = components.stream.vae.model.stream_encode(block_state.input)
        # Transpose latents
        latents = latents.transpose(2, 1)

        # Create generator from seed for reproducible generation
        frame_seed = block_state.base_seed + block_state.current_start
        rng = torch.Generator(device=latents.device).manual_seed(frame_seed)

        noise = torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=rng,
        )
        # Determine how noisy the latents should be
        noisy_latents = noise * block_state.noise_scale + latents * (1 - block_state.noise_scale)

        # Determine the number of denoising steps
        current_step = int(1000 * block_state.noise_scale) - 100

        block_state.latents = latents
        block_state.noisy_latents = noisy_latents
        block_state.current_step = current_step
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state


class StreamDiffusionV2DenoiseStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Denoise step that performs inference using the stream pipeline"

    @property
    def inputs(self) -> List[InputParam]:
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
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "denoised_pred",
                type_hint=torch.Tensor,
                description="Denoised prediction",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)

        # Use the stream's inference method
        denoised_pred = components.stream.inference(
            noise=block_state.noisy_latents,
            current_start=block_state.current_start,
            current_end=block_state.current_end,
            current_step=block_state.current_step,
            generator=block_state.generator,
        )

        block_state.denoised_pred = denoised_pred
        self.set_block_state(state, block_state)
        return components, state


class StreamDiffusionV2PostprocessStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Postprocess step that decodes denoised latents to pixel space"

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam(
                "denoised_pred",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [
            OutputParam(
                "output",
                type_hint=torch.Tensor,
                description="Decoded video frames",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components, state: PipelineState
    ) -> PipelineState:
        block_state = self.get_block_state(state)

        # Decode to pixel space
        output = components.stream.vae.stream_decode_to_pixel(block_state.denoised_pred)
        block_state.output = output

        self.set_block_state(state, block_state)
        return components, state


from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

VIDEO2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", StreamDiffusionV2TextEncoderStep),
        ("preprocess", StreamDiffusionV2PreprocessStep),
        ("denoise", StreamDiffusionV2DenoiseStep),
        ("postprocess", StreamDiffusionV2PostprocessStep),
    ]
)

ALL_BLOCKS = {
    "video2video": VIDEO2VIDEO_BLOCKS,
}


class StreamDiffusionV2Blocks(SequentialPipelineBlocks):
    block_classes = list(VIDEO2VIDEO_BLOCKS.copy().values())
    block_names = list(VIDEO2VIDEO_BLOCKS.copy().keys())
