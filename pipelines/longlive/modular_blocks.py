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
from .denoise import LongLiveDenoiseBlock

logger = diffusers_logging.get_logger(__name__)

# Define LongLive blocks - reuses most base blocks but has custom denoise
LONGLIVE_BLOCKS = InsertableDict(
    [
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("setup_kv_cache", SetupKVCacheBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", LongLiveDenoiseBlock),
        ("decode", DecodeBlock),
    ]
)

ALL_BLOCKS = LONGLIVE_BLOCKS


class LongLiveBlocks(SequentialPipelineBlocks):
    block_classes = list(LONGLIVE_BLOCKS.values())
    block_names = list(LONGLIVE_BLOCKS.keys())
