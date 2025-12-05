"""Modular blocks for Reward-Forcing inference pipeline.

Reward-Forcing uses few-step denoising (typically 4 steps) with
a specific timestep list for efficient video generation.

The blocks are identical to LongLive as they share the same architecture.
"""

from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..longlive.blocks import (
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
)
from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)

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
        (
            "set_transformer_blocks_local_attn_size",
            SetTransformerBlocksLocalAttnSizeBlock,
        ),
        ("auto_prepare_latents", AutoPrepareLatentsBlock),
        ("recache_frames", RecacheFramesBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_recache_frames", PrepareRecacheFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class RewardForcingBlocks(SequentialPipelineBlocks):
    """Sequential blocks for Reward-Forcing pipeline.

    Identical to LongLive blocks as they share the same causal architecture.
    The key difference is in the trained weights (4-step distilled model).
    """

    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
