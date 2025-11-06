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

from diffusers.utils import logging as diffusers_logging
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict

from ..base import (
    DenoiseBlock,
    DecodeBlock,
    SetTimestepsBlock,
    SetupKVCacheBlock,
    RecomputeKVCacheBlock,
    PrepareLatentsBlock,
    PrepareVideoLatentsBlock,
)
from .text_conditioning import TextConditioningBlock
from .before_denoise import BeforeDenoiseBlock
from .t2v_before_denoise import T2VBeforeDenoiseBlock
from .v2v_before_denoise import V2VBeforeDenoiseBlock

logger = diffusers_logging.get_logger(__name__)

# Define T2V (Text-to-Video) path blocks
T2V_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("before_denoise", BeforeDenoiseBlock),
        ("t2v_before_denoise", T2VBeforeDenoiseBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("setup_kv_cache", SetupKVCacheBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
    ]
)

# Define V2V (Video-to-Video) path blocks
V2V_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("before_denoise", BeforeDenoiseBlock),
        ("v2v_before_denoise", V2VBeforeDenoiseBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("setup_kv_cache", SetupKVCacheBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
    ]
)

# Main pipeline blocks - use T2V as default, but blocks will check trigger
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("before_denoise", BeforeDenoiseBlock),
        # Conditional blocks - they check block_trigger_input internally
        ("t2v_before_denoise", T2VBeforeDenoiseBlock),
        ("v2v_before_denoise", V2VBeforeDenoiseBlock),
        # Common blocks for both paths
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("setup_kv_cache", SetupKVCacheBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
    ]
)


class KreaRealtimeVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
