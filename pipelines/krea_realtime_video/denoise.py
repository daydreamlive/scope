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


class DenoiseBlock(ModularPipelineBlocks):
    model_name = "KreaRealtimeVideo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("scheduler", torch.nn.Module),
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Denoise block that performs iterative denoising"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
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
                "current_start",
                required=True,
                type_hint=int,
                description="Current start position",
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

        noise = block_state.latents
        denoising_step_list = block_state.timesteps
        num_frame_per_block = noise.shape[1]
        batch_size = noise.shape[0]

        # Set conditional dict for generator
        conditional_dict = {"prompt_embeds": block_state.prompt_embeds}
        components.stream.conditional_dict = conditional_dict

        # Calculate start_frame for kv_cache
        kv_cache_num_frames = getattr(components.stream, "kv_cache_num_frames", 3)
        start_frame = min(block_state.current_start, kv_cache_num_frames)

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            timestep = (
                torch.ones(
                    [batch_size, num_frame_per_block],
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
                    current_start=start_frame * components.stream.frame_seq_length,
                )
                next_timestep = denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=block_state.generator,
                )
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frame_per_block],
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
                    current_start=start_frame * components.stream.frame_seq_length,
                )

        # Store first context frame if this is the first iteration
        if block_state.current_start == 0:
            components.stream.first_context_frame = denoised_pred[:, :1]

        # Push the generated latents to the context frame buffer (sliding window)
        if components.stream.context_frame_buffer_max_size > 0:
            components.stream.context_frame_buffer = torch.cat(
                [
                    components.stream.context_frame_buffer,
                    denoised_pred.to(
                        components.stream.context_frame_buffer.device,
                        components.stream.context_frame_buffer.dtype,
                    ),
                ],
                dim=1,
            )[:, -components.stream.context_frame_buffer_max_size :]

        block_state.denoised_pred = denoised_pred

        self.set_block_state(state, block_state)
        return components, state
