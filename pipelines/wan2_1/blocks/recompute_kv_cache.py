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


class RecomputeKVCacheBlock(ModularPipelineBlocks):
    """Base Recompute KV Cache block that runs recompute across pipelines."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Base Recompute KV Cache block that runs recompute"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "current_start",
                type_hint=int,
                default=0,
                description="Current start position",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Recompute cache if current_start > 0 (not the first frame)
        if block_state.current_start > 0:
            components.stream._recompute_cache()

        self.set_block_state(state, block_state)
        return components, state
