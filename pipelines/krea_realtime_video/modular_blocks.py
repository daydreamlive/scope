from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    DecodeBlock,
    DenoiseBlock,
    PrepareLatentsBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks.prepare_context_frames import PrepareContextFramesBlock
from .blocks.recompute_kv_cache import RecomputeKVCacheBlock

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks for T2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("setup_caches", SetupCachesBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_context_frames", PrepareContextFramesBlock),
    ]
)


class KreaRealtimeVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
