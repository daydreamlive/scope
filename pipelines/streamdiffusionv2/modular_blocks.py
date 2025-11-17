from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import InsertableDict
from diffusers.utils import logging as diffusers_logging

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

logger = diffusers_logging.get_logger(__name__)

# Block sequences for StreamDiffusionV2
# Video mode: consumes input video and uses motion-aware noise control.
VIDEO_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("preprocess_video", PreprocessVideoBlock),
        ("noise_scale_controller", NoiseScaleControllerBlock),
        ("setup_caches", SetupCachesBlock),
        # PrepareLatentsBlock always runs to create base latents for text-to-video
        # workflows. For video-to-video workflows, PrepareVideoLatentsBlock will
        # override these latents using the encoded video.
        ("prepare_latents", PrepareLatentsBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)

# Text mode: pure text-to-video, no video-dependent blocks.
TEXT_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_latents", PrepareLatentsBlock),
        ("denoise", DenoiseBlock),
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class StreamDiffusionV2VideoBlocks(SequentialPipelineBlocks):
    block_classes = list(VIDEO_BLOCKS.values())
    block_names = list(VIDEO_BLOCKS.keys())


class StreamDiffusionV2TextBlocks(SequentialPipelineBlocks):
    block_classes = list(TEXT_BLOCKS.values())
    block_names = list(TEXT_BLOCKS.keys())
