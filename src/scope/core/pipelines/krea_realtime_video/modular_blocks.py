"""Single workflow for KreaRealtimeVideo pipeline using AutoPipelineBlocks for routing."""

from diffusers.modular_pipelines import SequentialPipelineBlocks

from ..multi_mode_blocks import ConfigureForModeBlock, LoadComponentsBlock
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
from .blocks import (
    AutoSetTransformerBlocksLocalAttnSizeBlock,
    PrepareContextFramesBlock,
    RecomputeKVCacheBlock,
)


class KreaRealtimeVideoWorkflow(SequentialPipelineBlocks):
    """Single workflow for KreaRealtimeVideo supporting both T2V and V2V.

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
        # Local attention size (AUTO-ROUTED: V2V only)
        AutoSetTransformerBlocksLocalAttnSizeBlock,
        # Latent preparation (AUTO-ROUTED: T2V vs V2V, after cache setup)
        AutoPrepareLatentsBlock,
        # KV cache management (shared)
        RecomputeKVCacheBlock,
        # Generation (shared)
        DenoiseBlock,
        DecodeBlock,
        # Context frame preparation (shared)
        PrepareContextFramesBlock,
        # Preparation for next iteration (shared)
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
        "recompute_kv_cache",
        "denoise",
        "decode",
        "prepare_context_frames",
        "prepare_next",
    ]
