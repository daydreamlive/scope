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


class PrepareVideoLatentsBlock(ModularPipelineBlocks):
    """Base Prepare Video Latents block that generates noisy latents based on encoded latents for video-to-video across pipelines."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Base Prepare Video Latents block that generates noisy latents based on encoded latents for video-to-video"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                type_hint=torch.Tensor,
                required=True,
                description="Encoded latents from video frames",
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
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Scale of noise to add to latents",
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
                description="Generated noisy latents based on encoded latents",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Create generator from seed for reproducible generation
        frame_seed = block_state.base_seed + block_state.current_start
        rng = torch.Generator(device=block_state.latents.device).manual_seed(frame_seed)

        # Add noise to latents
        noise = torch.randn(
            block_state.latents.shape,
            device=block_state.latents.device,
            dtype=block_state.latents.dtype,
            generator=rng,
        )
        # Determine how noisy the latents should be
        noisy_latents = noise * block_state.noise_scale + block_state.latents * (
            1 - block_state.noise_scale
        )

        block_state.latents = noisy_latents
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
