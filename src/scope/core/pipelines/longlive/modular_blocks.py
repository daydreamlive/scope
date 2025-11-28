"""Single workflow for LongLive pipeline using AutoPipelineBlocks for routing."""

from diffusers.modular_pipelines import SequentialPipelineBlocks

from ..multi_mode_blocks import ConfigureForModeBlock, LoadComponentsBlock
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
from .blocks import (
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
)


class LongLiveWorkflow(SequentialPipelineBlocks):
    """Single workflow for LongLive supporting both T2V and V2V.

    Uses two AutoPipelineBlocks for routing:
    1. AutoPreprocessVideoBlock - routes video preprocessing before cache setup
    2. AutoPrepareLatentsBlock - routes latent preparation after cache setup

    This maintains the correct execution order where video preprocessing happens
    before SetupCachesBlock, but latent encoding happens after.
    """

    block_classes = [
        # Configuration
        ConfigureForModeBlock,
        LoadComponentsBlock,
        # Text conditioning (shared)
        TextConditioningBlock,
        EmbeddingBlendingBlock,
        SetTimestepsBlock,
        # Video preprocessing (AUTO-ROUTED: V2V only, before cache setup)
        AutoPreprocessVideoBlock,
        # Setup (shared)
        SetupCachesBlock,
        SetTransformerBlocksLocalAttnSizeBlock,
        # Latent preparation (AUTO-ROUTED: T2V vs V2V, after cache setup)
        AutoPrepareLatentsBlock,
        # Frame management (shared)
        RecacheFramesBlock,
        # Generation (shared)
        DenoiseBlock,
        CleanKVCacheBlock,
        DecodeBlock,
        # Preparation for next iteration (shared)
        PrepareRecacheFramesBlock,
        PrepareNextBlock,
    ]

    block_names = [
        "configure_for_mode",
        "load_components",
        "text_conditioning",
        "embedding_blending",
        "set_timesteps",
        "auto_preprocess_video",
        "setup_caches",
        "set_transformer_blocks_local_attn_size",
        "auto_prepare_latents",
        "recache_frames",
        "denoise",
        "clean_kv_cache",
        "decode",
        "prepare_recache_frames",
        "prepare_next",
    ]
