from typing import Any

import torch
from diffusers.modular_pipelines import PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    InputParam,
    InsertableDict,
)
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    CleanKVCacheBlock,
    DecodeBlock,
    EmbeddingBlendingBlock,
    NoiseScaleControllerBlock,
    PrepareNextBlock,
    PrepareVideoLatentsBlock,
    PreprocessVideoBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from ..wan2_1.blocks import (
    DenoiseBlock as BaseDenoiseBlock,
)

logger = diffusers_logging.get_logger(__name__)


# Main pipeline blocks for V2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("preprocess_video", PreprocessVideoBlock),
        ("noise_scale_controller", NoiseScaleControllerBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("denoise", BaseDenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class StreamDiffusionV2Blocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
