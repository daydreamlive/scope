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

from einops import rearrange


class EncodeVideoBlock(ModularPipelineBlocks):
    """Base Encode Video block that encodes video frames to latents using VAE across pipelines."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Base Encode Video block that encodes video frames to latents using VAE"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "block_trigger_input",
                type_hint=str,
                description="Trigger input to determine if this block should execute",
            ),
            InputParam(
                "video_tensor",
                type_hint=torch.Tensor,
                required=False,
                description="Input video tensor (N frames)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Encoded latents from video frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        # Check trigger first before validating required inputs
        block_trigger = state.values.get("block_trigger_input")
        if block_trigger != "v2v":
            # Skip if not V2V path - return state as-is
            return components, state

        block_state = self.get_block_state(state)

        # Validate video_tensor is present for V2V path
        if not hasattr(block_state, "video_tensor") or block_state.video_tensor is None:
            raise ValueError("video_tensor is required for V2V path")

        # Encode video frames to latents using VAE
        # video_tensor is expected to be in BCTHW format
        latents = components.vae.encode_to_latent(
            rearrange(block_state.video_tensor, "B T C H W -> B C T H W")
        )

        block_state.latents = latents

        self.set_block_state(state, block_state)
        return components, state
