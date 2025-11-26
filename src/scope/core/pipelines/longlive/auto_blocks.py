"""AutoPipelineBlocks and workflow definitions for LongLive pipeline.

This module defines the multi-mode workflows for LongLive pipeline using
the new declarative architecture pattern.
"""

from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    SequentialPipelineBlocks,
)

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
from .blocks import (
    PrepareRecacheFramesBlock,
    RecacheFramesBlock,
    SetTransformerBlocksLocalAttnSizeBlock,
)


class LongLiveTextWorkflow(SequentialPipelineBlocks):
    """Complete workflow for LongLive text-to-video mode.

    This workflow implements pure text-to-video generation with long-form
    generation capabilities. It uses:
    - 4 denoising steps (configurable)
    - Pure noise latents initialization
    - Frame recaching for long-form consistency
    """

    block_classes = [
        # Configuration (automatic mode detection and component loading)
        ConfigureForModeBlock,
        LoadComponentsBlock,
        # Conditioning (shared with video mode)
        TextConditioningBlock,
        EmbeddingBlendingBlock,
        SetTimestepsBlock,
        # Setup
        SetupCachesBlock,
        SetTransformerBlocksLocalAttnSizeBlock,
        # Text-specific latent preparation
        PrepareLatentsBlock,
        # Frame management for long-form generation
        RecacheFramesBlock,
        # Core generation
        DenoiseBlock,
        CleanKVCacheBlock,
        DecodeBlock,
        # Preparation for next iteration
        PrepareRecacheFramesBlock,
        PrepareNextBlock,
    ]

    block_names = [
        "configure_for_mode",
        "load_components",
        "text_conditioning",
        "embedding_blending",
        "set_timesteps",
        "setup_caches",
        "set_transformer_blocks_local_attn_size",
        "prepare_latents",
        "recache_frames",
        "denoise",
        "clean_kv_cache",
        "decode",
        "prepare_recache_frames",
        "prepare_next",
    ]


class LongLiveVideoWorkflow(SequentialPipelineBlocks):
    """Complete workflow for LongLive video-to-video mode.

    This workflow implements video-to-video generation with motion-aware
    processing. It uses:
    - 2 denoising steps (configurable)
    - Video encoding to latent space
    - Noise scale controller for motion-aware noise injection
    - Frame recaching for temporal consistency
    """

    block_classes = [
        # Configuration (automatic mode detection and component loading)
        ConfigureForModeBlock,
        LoadComponentsBlock,
        # Conditioning (shared with text mode)
        TextConditioningBlock,
        EmbeddingBlendingBlock,
        SetTimestepsBlock,
        # Video-specific preprocessing
        PreprocessVideoBlock,
        NoiseScaleControllerBlock,
        # Setup
        SetupCachesBlock,
        SetTransformerBlocksLocalAttnSizeBlock,
        # Video-specific latent preparation
        PrepareVideoLatentsBlock,
        # Frame management for temporal consistency
        RecacheFramesBlock,
        # Core generation
        DenoiseBlock,
        CleanKVCacheBlock,
        DecodeBlock,
        # Preparation for next iteration
        PrepareRecacheFramesBlock,
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
        "set_transformer_blocks_local_attn_size",
        "prepare_video_latents",
        "recache_frames",
        "denoise",
        "clean_kv_cache",
        "decode",
        "prepare_recache_frames",
        "prepare_next",
    ]


class LongLiveAutoBlocks(AutoPipelineBlocks):
    """AutoPipelineBlocks for LongLive pipeline.

    Routes to appropriate workflow based on input presence:
    - Video mode: Triggered when 'video' input is present
    - Text mode: Default when no video input

    The mode is actually determined by the generation_mode parameter
    set in state, but AutoPipelineBlocks validates that the expected
    inputs are present.
    """

    block_classes = [
        LongLiveVideoWorkflow,
        LongLiveTextWorkflow,
    ]

    block_names = ["video_mode", "text_mode"]

    # Trigger inputs: video mode checks for "video", text mode has no requirements
    block_trigger_inputs = ["video", None]

    @property
    def description(self) -> str:
        """Return description of auto blocks behavior."""
        return (
            "LongLive auto blocks for text-to-video and video-to-video.\n"
            "- Video mode runs when 'video' input is provided (2 denoising steps).\n"
            "- Text mode runs when no 'video' input is provided (4 denoising steps)."
        )
