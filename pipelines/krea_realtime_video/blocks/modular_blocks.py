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

from .after_decode import AfterDecodeBlock
from .decode import DecodeBlock
from .denoise import DenoiseBlock
from .prepare_latents import PrepareLatentsBlock
from .recompute_kv_cache import RecomputeKVCacheBlock
from .set_timesteps import SetTimestepsBlock
from .setup_kv_cache import SetupKVCacheBlock
from .text_conditioning import TextConditioningBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks for T2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
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
