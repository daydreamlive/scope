"""Single workflow for KreaRealtimeVideo pipeline using AutoPrepareLatentsBlock."""

from diffusers.modular_pipelines import SequentialPipelineBlocks

from ..multi_mode_blocks import ConfigureForModeBlock, LoadComponentsBlock
from ..wan2_1.blocks import (
    AutoPrepareLatentsBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    PrepareNextBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from .blocks import PrepareContextFramesBlock, RecomputeKVCacheBlock


class KreaRealtimeVideoWorkflow(SequentialPipelineBlocks):
    """Single workflow for KreaRealtimeVideo supporting both T2V and V2V.

    Uses AutoPrepareLatentsBlock for automatic routing between text-to-video
    and video-to-video latent preparation. All shared blocks appear once.
    """

    block_classes = [
        # Configuration
        ConfigureForModeBlock,
        LoadComponentsBlock,
        # Text conditioning (shared)
        TextConditioningBlock,
        EmbeddingBlendingBlock,
        SetTimestepsBlock,
        # Latent preparation (AUTO-ROUTED: T2V vs V2V)
        AutoPrepareLatentsBlock,
        # Setup (shared)
        SetupCachesBlock,
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
        "auto_prepare_latents",
        "setup_caches",
        "recompute_kv_cache",
        "denoise",
        "decode",
        "prepare_context_frames",
        "prepare_next",
    ]
