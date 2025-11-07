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


class LongLiveDenoiseBlock(ModularPipelineBlocks):
    """LongLive-specific Denoise block that performs iterative denoising with context frame rerun."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("scheduler", torch.nn.Module),
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "LongLive Denoise block that performs iterative denoising with context frame rerun"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "timesteps",
                required=True,
                type_hint=torch.Tensor,
                description="Timesteps for denoising",
            ),
            InputParam(
                "prompt_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Prompt embeddings for conditioning",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                default=42,
                description="Base seed for random number generation",
            ),
            InputParam(
                "current_start",
                required=True,
                type_hint=int,
                description="Current start position",
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

        denoising_step_list = block_state.timesteps

        latent_height = block_state.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = block_state.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        generator_param = next(components.generator.model.parameters())

        # Create generator from seed for reproducible generation
        # Derive unique seed per block of latents using current_start as offset
        frame_seed = block_state.base_seed + block_state.current_start
        rng = torch.Generator(device=generator_param.device).manual_seed(frame_seed)

        noise = torch.randn(
            [
                1,  # batch_size (LongLive always uses batch_size=1)
                block_state.num_frame_per_block,
                16,
                latent_height,
                latent_width,
            ],
            device=generator_param.device,
            dtype=generator_param.dtype,
            generator=rng,
        )

        # Set conditional dict for generator
        conditional_dict = {"prompt_embeds": block_state.prompt_embeds}
        components.stream.conditional_dict = conditional_dict

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            timestep = (
                torch.ones(
                    [1, block_state.num_frame_per_block],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(denoising_step_list) - 1:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=components.stream.kv_cache1,
                    crossattn_cache=components.stream.crossattn_cache,
                    current_start=block_state.current_start * components.stream.frame_seq_length,
                )
                next_timestep = denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=rng,
                )
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [1 * block_state.num_frame_per_block],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=components.stream.kv_cache1,
                    crossattn_cache=components.stream.crossattn_cache,
                    current_start=block_state.current_start * components.stream.frame_seq_length,
                )

        # LongLive specific: rerun with clean context to update cache
        context_timestep = torch.ones_like(timestep) * 0
        components.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=components.stream.kv_cache1,
            crossattn_cache=components.stream.crossattn_cache,
            current_start=block_state.current_start * components.stream.frame_seq_length,
        )

        # Push the generated latents to the recache buffer (sliding window)
        # Shift buffer left, append new frames at end
        components.stream.recache_buffer = torch.cat(
            [
                components.stream.recache_buffer[:, block_state.num_frame_per_block :],
                denoised_pred.to(components.stream.recache_buffer.device),
            ],
            dim=1,
        )

        block_state.denoised_pred = denoised_pred

        self.set_block_state(state, block_state)
        return components, state
