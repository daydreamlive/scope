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
    AutoPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

from .encode_video import EncodeVideoBlock


class PassThroughBlock(ModularPipelineBlocks):
    """Pass-through block for T2V path when no video encoding is needed."""

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return []

    @property
    def description(self) -> str:
        return "Pass-through block for T2V path"

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        # Pass through without modification
        return components, state


class EncodeVideoAutoBlocks(AutoPipelineBlocks):
    """AutoPipelineBlocks that routes to EncodeVideoBlock or PassThroughBlock based on video_tensor input."""

    block_classes = [EncodeVideoBlock, PassThroughBlock]
    block_names = ["encode_video", "pass_through"]
    # Trigger based on video_tensor: if video_tensor is provided, use EncodeVideoBlock, otherwise PassThroughBlock (default)
    block_trigger_inputs = ["video_tensor", None]

    @property
    def description(self) -> str:
        return (
            "AutoPipelineBlocks that routes to encode video blocks:\n"
            "- EncodeVideoBlock is triggered when 'video_tensor' is provided (V2V path).\n"
            "- PassThroughBlock is the default when 'video_tensor' is not provided (T2V path)."
        )
