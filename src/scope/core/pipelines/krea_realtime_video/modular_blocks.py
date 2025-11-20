from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareLatentsBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import PrepareContextFramesBlock, RecomputeKVCacheBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks for T2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_latents", PrepareLatentsBlock),
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
