from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
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
from .blocks import PrepareContextFramesBlock, RecomputeKVCacheBlock

logger = diffusers_logging.get_logger(__name__)

# Block sequences for KreaRealtimeVideo
# Text mode: pure text-to-video with fast generation
TEXT_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_context_frames", PrepareContextFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)

# Video mode: video-to-video with motion-aware processing
VIDEO_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("preprocess_video", PreprocessVideoBlock),
        ("noise_scale_controller", NoiseScaleControllerBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("recompute_kv_cache", RecomputeKVCacheBlock),
        ("denoise", DenoiseBlock),
        ("decode", DecodeBlock),
        ("prepare_context_frames", PrepareContextFramesBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class KreaRealtimeVideoTextBlocks(SequentialPipelineBlocks):
    block_classes = list(TEXT_BLOCKS.values())
    block_names = list(TEXT_BLOCKS.keys())


class KreaRealtimeVideoVideoBlocks(SequentialPipelineBlocks):
    block_classes = list(VIDEO_BLOCKS.values())
    block_names = list(VIDEO_BLOCKS.keys())
