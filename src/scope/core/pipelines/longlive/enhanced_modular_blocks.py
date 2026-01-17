# Enhanced modular blocks with FreSca and TSR support
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    AutoPreprocessVideoBlock,
    CleanKVCacheBlock,
    DecodeBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from ..wan2_1.vace.blocks import VaceEncodingBlock
from .blocks import (
    EnhancedDenoiseBlock,
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
)

logger = diffusers_logging.get_logger(__name__)

# Enhanced pipeline blocks with FreSca and TSR support
# Uses EnhancedDenoiseBlock instead of standard DenoiseBlock
ENHANCED_BLOCKS = InsertableDict(
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
        ("recache_frames", RecacheFramesBlock),
        ("vace_encoding", VaceEncodingBlock),
        ("denoise", EnhancedDenoiseBlock),  # Enhanced version
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_recache_frames", PrepareRecacheFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class EnhancedLongLiveBlocks(SequentialPipelineBlocks):
    """
    Enhanced LongLive blocks with FreSca and TSR support.

    Enhancement parameters can be passed via pipeline __call__:
    - enable_fresca: Enable frequency-selective scaling
    - fresca_scale_low: Low-frequency scaling (default 1.0)
    - fresca_scale_high: High-frequency scaling (default 1.15)
    - fresca_freq_cutoff: Frequency cutoff radius (default 20)
    - fresca_adaptive: Step-adaptive scaling (default False)
    - enable_tsr: Enable temporal score rescaling
    - tsr_k: TSR sampling temperature (default 0.95)
    - tsr_sigma: TSR SNR influence factor (default 0.1)
    """

    block_classes = list(ENHANCED_BLOCKS.values())
    block_names = list(ENHANCED_BLOCKS.keys())
