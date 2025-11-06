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


class SetTimestepsBlock(ModularPipelineBlocks):
    model_name = "KreaRealtimeVideo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Set Timesteps block that configures denoising steps based on scheduler"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "denoising_step_list",
                type_hint=torch.Tensor,
                description="List of denoising steps",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "timesteps",
                type_hint=torch.Tensor,
                description="Configured timesteps for denoising",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # The timesteps are already configured in the scheduler via denoising_step_list
        # This block just ensures they're available in state
        if hasattr(block_state, "denoising_step_list") and block_state.denoising_step_list is not None:
            block_state.timesteps = block_state.denoising_step_list
        else:
            # Default timesteps if not provided
            block_state.timesteps = components.scheduler.timesteps

        self.set_block_state(state, block_state)
        return components, state
