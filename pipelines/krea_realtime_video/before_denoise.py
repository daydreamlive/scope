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


class BeforeDenoiseBlock(ModularPipelineBlocks):
    model_name = "KreaRealtimeVideo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Before Denoise block that routes to T2V or V2V based on input video tensor"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video_tensor",
                type_hint=torch.Tensor,
                description="Optional input video tensor (N frames) for video-to-video",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "block_trigger_input",
                type_hint=str,
                description="Trigger to pick which path: 't2v' or 'v2v'",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Determine which path to take based on whether video tensor is provided
        if hasattr(block_state, "video_tensor") and block_state.video_tensor is not None:
            block_state.block_trigger_input = "v2v"
        else:
            block_state.block_trigger_input = "t2v"

        self.set_block_state(state, block_state)
        return components, state
