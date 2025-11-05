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


class StreamDiffusionV2PreprocessStep(ModularPipelineBlocks):
    model_name = "StreamDiffusionV2"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Preprocess step that encodes input frames to latents and adds noise"

    @property
    def inputs(self) -> list[InputParam]:
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
    def intermediate_outputs(self) -> list[OutputParam]:
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
    def __call__(self, components, state: PipelineState) -> PipelineState:
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
        noisy_latents = noise * block_state.noise_scale + latents * (
            1 - block_state.noise_scale
        )

        # Determine the number of denoising steps
        current_step = int(1000 * block_state.noise_scale) - 100

        # Update state directly without intermediate variables where possible
        block_state.latents = latents
        block_state.noisy_latents = noisy_latents
        block_state.current_step = current_step
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
