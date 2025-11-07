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

from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ...wan2_1.blocks import (
    DecodeBlock,
    DenoiseBlock,
    RecomputeKVCacheBlock,
    SetTimestepsBlock,
    SetupKVCacheBlock,
)
from ...wan2_1.blocks.encode_video_auto import EncodeVideoAutoBlocks
from ...wan2_1.blocks.prepare_latents_auto import PrepareLatentsAutoBlocks
from .after_decode import AfterDecodeBlock
from .before_denoise import BeforeDenoiseBlock
from .before_denoise_auto import BeforeDenoiseAutoBlocks
from .text_conditioning import TextConditioningBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks - use AutoPipelineBlocks for conditional routing
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("before_denoise", BeforeDenoiseBlock),
        # AutoPipelineBlocks that routes to T2V or V2V based on video_tensor
        ("before_denoise_auto", BeforeDenoiseAutoBlocks),
        # AutoPipelineBlocks that routes to EncodeVideoBlock or PassThroughBlock based on video_tensor
        ("encode_video", EncodeVideoAutoBlocks),
        # Common blocks for both paths
        ("set_timesteps", SetTimestepsBlock),
        # AutoPipelineBlocks that routes to PrepareLatentsBlock or PrepareVideoLatentsBlock based on latents
        ("prepare_latents", PrepareLatentsAutoBlocks),
        ("setup_kv_cache", SetupKVCacheBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("after_decode", AfterDecodeBlock),
    ]
)


class KreaRealtimeVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
