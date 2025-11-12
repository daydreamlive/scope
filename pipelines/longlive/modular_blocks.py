from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    DecodeBlock,
    DenoiseBlock,
    PrepareLatentsBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import (
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
)

logger = diffusers_logging.get_logger(__name__)

# Main pipeline blocks for T2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("setup_caches", SetupCachesBlock),
        (
            "set_transformer_blocks_local_attn_size",
            SetTransformerBlocksLocalAttnSizeBlock,
        ),
        ("recache_frames", RecacheFramesBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_recache_frames", PrepareRecacheFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class LongLiveBlocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
