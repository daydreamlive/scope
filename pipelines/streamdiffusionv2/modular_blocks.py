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

from .preprocess import StreamDiffusionV2PreprocessStep
from .decoders import StreamDiffusionV2PostprocessStep
from .encoders import StreamDiffusionV2TextEncoderStep
from .denoise import StreamDiffusionV2DenoiseStep

logger = diffusers_logging.get_logger(__name__)

VIDEO2VIDEO_BLOCKS = InsertableDict(
    [
        ("text_encoder", StreamDiffusionV2TextEncoderStep),
        ("preprocess", StreamDiffusionV2PreprocessStep),
        ("denoise", StreamDiffusionV2DenoiseStep),
        ("postprocess", StreamDiffusionV2PostprocessStep),
    ]
)

ALL_BLOCKS = {
    "video2video": VIDEO2VIDEO_BLOCKS,
}


class StreamDiffusionV2Blocks(SequentialPipelineBlocks):
    block_classes = list(VIDEO2VIDEO_BLOCKS.values())
    block_names = list(VIDEO2VIDEO_BLOCKS.keys())
