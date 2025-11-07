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

from .t2v_before_denoise import T2VBeforeDenoiseBlock
from .v2v_before_denoise import V2VBeforeDenoiseBlock


class BeforeDenoiseAutoBlocks(AutoPipelineBlocks):
    """AutoPipelineBlocks that routes to T2V or V2V before_denoise blocks based on video_tensor input."""

    block_classes = [V2VBeforeDenoiseBlock, T2VBeforeDenoiseBlock]
    block_names = ["v2v", "t2v"]
    # Trigger based on video_tensor: if video_tensor is provided, use V2V, otherwise T2V (default)
    block_trigger_inputs = ["video_tensor", None]

    @property
    def description(self) -> str:
        return (
            "AutoPipelineBlocks that routes to T2V or V2V before_denoise blocks:\n"
            "- V2V workflow is triggered when 'video_tensor' is provided.\n"
            "- T2V workflow is the default when 'video_tensor' is not provided."
        )
