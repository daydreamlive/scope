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

from diffusers.modular_pipelines import AutoPipelineBlocks

from .prepare_latents import PrepareLatentsBlock
from .prepare_video_latents import PrepareVideoLatentsBlock


class PrepareLatentsAutoBlocks(AutoPipelineBlocks):
    """AutoPipelineBlocks that routes to PrepareLatentsBlock or PrepareVideoLatentsBlock based on latents input."""

    block_classes = [PrepareVideoLatentsBlock, PrepareLatentsBlock]
    block_names = ["prepare_video_latents", "prepare_latents"]
    # Trigger based on latents: if latents is provided, use PrepareVideoLatentsBlock, otherwise PrepareLatentsBlock (default)
    block_trigger_inputs = ["latents", None]

    @property
    def description(self) -> str:
        return (
            "AutoPipelineBlocks that routes to prepare latents blocks:\n"
            "- PrepareVideoLatentsBlock is triggered when 'latents' is provided (V2V path).\n"
            "- PrepareLatentsBlock is the default when 'latents' is not provided (T2V path)."
        )
