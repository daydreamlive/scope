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

# The VAE does 8x spatial downsampling
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 8


class PrepareLatentsBlock(ModularPipelineBlocks):
    """Base Prepare Latents block that generates empty latents (noise) for text-to-video across pipelines."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Base Prepare Latents block that generates empty latents (noise) for text-to-video"

    @property
    def inputs(self) -> list[InputParam]:
        return [
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
            InputParam(
                "num_frame_per_block",
                type_hint=int,
                default=1,
                description="Number of frames per block",
            ),
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
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
                "latents",
                type_hint=torch.Tensor,
                description="Generated empty latents (noise)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        generator_param = next(components.generator.model.parameters())
        latent_height = block_state.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = block_state.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR

        # Create generator from seed for reproducible generation
        frame_seed = block_state.base_seed + block_state.current_start
        rng = torch.Generator(device=generator_param.device).manual_seed(frame_seed)

        # Generate empty latents (noise)
        latents = torch.randn(
            [
                1,  # batch_size
                block_state.num_frame_per_block,
                16,
                latent_height,
                latent_width,
            ],
            device=generator_param.device,
            dtype=generator_param.dtype,
            generator=rng,
        )

        block_state.latents = latents
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
