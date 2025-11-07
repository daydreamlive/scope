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
)


class T2VBeforeDenoiseBlock(ModularPipelineBlocks):
    model_name = "KreaRealtimeVideo"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "T2V Before Denoise block - default path for text-to-video when no input video is provided"

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        # This block is a pass-through for T2V path
        # The actual work is done in subsequent blocks (Set Timesteps, Prepare Latents, etc.)
        self.set_block_state(state, block_state)
        return components, state
