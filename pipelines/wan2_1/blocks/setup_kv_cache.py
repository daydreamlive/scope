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


class SetupKVCacheBlock(ModularPipelineBlocks):
    """Base Setup KV Cache block that ensures caches are setup and optionally re-initialized across pipelines."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("stream", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Base Setup KV Cache block that makes sure caches are setup and optionally re-initialized"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "init_cache",
                type_hint=bool,
                default=False,
                description="Whether to initialize cache",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # The stream object manages its own cache initialization
        # This block ensures caches are setup if needed
        if block_state.init_cache:
            generator_param = next(components.stream.generator.model.parameters())
            components.stream._initialize_kv_cache(
                batch_size=components.stream.batch_size,
                dtype=generator_param.dtype,
                device=generator_param.device,
            )
            components.stream._initialize_crossattn_cache(
                batch_size=components.stream.batch_size,
                dtype=generator_param.dtype,
                device=generator_param.device,
            )

        self.set_block_state(state, block_state)
        return components, state
