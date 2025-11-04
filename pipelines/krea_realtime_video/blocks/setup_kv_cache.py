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
            ComponentSpec("generator", torch.nn.Module),
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
            InputParam(
                "batch_size",
                type_hint=int,
                default=1,
                description="Batch size for cache initialization",
            ),
            InputParam(
                "local_attn_size",
                type_hint=int | None,
                default=None,
                description="Local attention size (optional, for cache size calculation)",
            ),
            InputParam(
                "frame_seq_length",
                type_hint=int | None,
                default=None,
                description="Frame sequence length (optional, for cache size calculation)",
            ),
            InputParam(
                "kv_cache_size_override",
                type_hint=int | None,
                default=None,
                description="Override for KV cache size (optional)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        # Initialize caches if needed
        if block_state.init_cache:
            generator_param = next(components.generator.model.parameters())
            components.generator.initialize_kv_cache(
                batch_size=block_state.batch_size,
                dtype=generator_param.dtype,
                device=generator_param.device,
                kv_cache_size_override=block_state.kv_cache_size_override,
                local_attn_size=block_state.local_attn_size,
                frame_seq_length=block_state.frame_seq_length,
            )
            components.generator.initialize_crossattn_cache(
                batch_size=block_state.batch_size,
                dtype=generator_param.dtype,
                device=generator_param.device,
            )

        self.set_block_state(state, block_state)
        return components, state
