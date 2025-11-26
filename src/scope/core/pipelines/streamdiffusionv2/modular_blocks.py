"""Unified workflow for StreamDiffusionV2 pipeline.

This module defines a single unified workflow for StreamDiffusionV2 pipeline that
conditionally executes blocks based on input presence. This aligns with
the original diffusers modular pipeline design philosophy where blocks
handle conditional execution internally rather than using separate workflows.
"""

from diffusers.modular_pipelines import SequentialPipelineBlocks

from ..multi_mode_blocks import ConfigureForModeBlock, LoadComponentsBlock
from ..wan2_1.blocks import (
    CleanKVCacheBlock,
    DecodeBlock,
    DenoiseBlock,
    EmbeddingBlendingBlock,
    NoiseScaleControllerBlock,
    PrepareLatentsBlock,
    PrepareNextBlock,
    PrepareVideoLatentsBlock,
    PreprocessVideoBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)


class StreamDiffusionV2UnifiedWorkflow(SequentialPipelineBlocks):
    """Unified workflow for StreamDiffusionV2 supporting both text-to-video and video-to-video.

    This workflow uses conditional block execution to support both modes in a single
    block graph, eliminating the need for separate workflows. Blocks self-determine
    whether to execute based on input presence:

    Text-to-video path:
    - PrepareLatentsBlock generates pure noise latents
    - Video-specific blocks (PreprocessVideo, NoiseScaleController, PrepareVideoLatents) skip

    Video-to-video path:
    - PreprocessVideoBlock preprocesses input video
    - NoiseScaleControllerBlock adjusts noise based on motion
    - PrepareVideoLatentsBlock encodes video to noisy latents
    - PrepareLatentsBlock skips

    This design aligns with the original diffusers modular pipeline philosophy where
    the block graph structure is shared across modes, with conditional execution
    determined by input availability rather than separate workflow routing.
    """

    block_classes = [
        # Configuration and component loading
        ConfigureForModeBlock,
        LoadComponentsBlock,
        # Text conditioning (shared across modes)
        TextConditioningBlock,
        EmbeddingBlendingBlock,
        SetTimestepsBlock,
        # Video preprocessing (skips if no video input)
        PreprocessVideoBlock,
        NoiseScaleControllerBlock,
        # Setup (shared across modes)
        SetupCachesBlock,
        # Latent preparation (one skips based on video presence)
        PrepareLatentsBlock,
        PrepareVideoLatentsBlock,
        # Core generation (shared across modes)
        DenoiseBlock,
        CleanKVCacheBlock,
        DecodeBlock,
        # Preparation for next iteration
        PrepareNextBlock,
    ]

    block_names = [
        "configure_for_mode",
        "load_components",
        "text_conditioning",
        "embedding_blending",
        "set_timesteps",
        "preprocess_video",
        "noise_scale_controller",
        "setup_caches",
        "prepare_latents",
        "prepare_video_latents",
        "denoise",
        "clean_kv_cache",
        "decode",
        "prepare_next",
    ]
