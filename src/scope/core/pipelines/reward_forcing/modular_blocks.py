from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    SetTimestepsBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from ..wan2_1.vace.blocks import VaceEncodingBlock
from .blocks import PrepareNextBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks with multi-mode support (text-to-video and video-to-video)
# AutoPreprocessVideoBlock: Routes to video preprocessing when 'video' input provided
# AutoPrepareLatentsBlock: Routes to PrepareVideoLatentsBlock or PrepareLatentsBlock
# VaceEncodingBlock: Encodes VACE context for conditioning
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        (
            "set_transformer_blocks_local_attn_size",
            SetTransformerBlocksLocalAttnSizeBlock,
        ),
        ("auto_preprocess_video", AutoPreprocessVideoBlock),
        ("auto_prepare_latents", AutoPrepareLatentsBlock),
        ("vace_encoding", VaceEncodingBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class RewardForcingBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
