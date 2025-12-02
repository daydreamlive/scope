from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import PrepareContextFramesBlock, RecomputeKVCacheBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks with multi-mode support (text-to-video and video-to-video)
# AutoPreprocessVideoBlock: Routes to video preprocessing when 'video' input provided
# AutoPrepareLatentsBlock: Routes to PrepareVideoLatentsBlock or PrepareLatentsBlock
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("auto_preprocess_video", AutoPreprocessVideoBlock),
        ("setup_caches", SetupCachesBlock),
        ("auto_prepare_latents", AutoPrepareLatentsBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_context_frames", PrepareContextFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class KreaRealtimeVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
