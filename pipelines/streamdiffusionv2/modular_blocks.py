from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    PrepareNextBlock,
    PrepareVideoLatentsBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks for V2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        # TODO: NoiseScaleControllerBlock
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("setup_caches", SetupCachesBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class StreamDiffusionV2Blocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
