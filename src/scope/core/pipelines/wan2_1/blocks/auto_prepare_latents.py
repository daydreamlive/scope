"""AutoPipelineBlocks for video preprocessing and latent preparation routing.

This module provides block-level routing between text-to-video and video-to-video
workflows at two critical points:
1. Video preprocessing (before SetupCachesBlock)
2. Latent preparation (after SetupCachesBlock)

This split maintains the correct execution order from the original unified workflow.
"""

from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    SequentialPipelineBlocks,
)

from .noise_scale_controller import NoiseScaleControllerBlock
from .prepare_latents import PrepareLatentsBlock
from .prepare_video_latents import PrepareVideoLatentsBlock
from .preprocess_video import PreprocessVideoBlock


class VideoPreprocessingWorkflow(SequentialPipelineBlocks):
    """Video preprocessing workflow for V2V mode.

    Preprocesses input video and applies motion-aware noise control.
    Runs BEFORE SetupCachesBlock.
    """

    block_classes = [
        PreprocessVideoBlock,
        NoiseScaleControllerBlock,
    ]

    block_names = [
        "preprocess_video",
        "noise_scale_controller",
    ]


class AutoPreprocessVideoBlock(AutoPipelineBlocks):
    """Auto-routing block for video preprocessing.

    Routes to video preprocessing workflow when 'video' input is provided,
    otherwise skips (AutoPipelineBlocks auto-skips when no trigger matches).
    This runs BEFORE SetupCachesBlock.
    """

    block_classes = [
        VideoPreprocessingWorkflow,
    ]

    block_names = [
        "video_preprocessing",
    ]

    block_trigger_inputs = [
        "video",
    ]

    @property
    def description(self):
        return (
            "AutoPreprocessVideoBlock: Routes video preprocessing before cache setup:\n"
            " - Routes to video preprocessing when 'video' input is provided\n"
            " - Skips preprocessing when no 'video' input is provided\n"
        )


class AutoPrepareLatentsBlock(AutoPipelineBlocks):
    """Auto-routing block for latent preparation.

    Routes between text-to-video and video-to-video latent preparation
    based on whether 'video' input is provided. This runs AFTER SetupCachesBlock.

    Uses blocks directly instead of wrapper workflows since each path is a single block.
    """

    block_classes = [
        PrepareVideoLatentsBlock,
        PrepareLatentsBlock,
    ]

    block_names = [
        "prepare_video_latents",
        "prepare_latents",
    ]

    block_trigger_inputs = [
        "video",
        None,
    ]

    @property
    def description(self):
        return (
            "AutoPrepareLatentsBlock: Routes latent preparation after cache setup:\n"
            " - Routes to PrepareVideoLatentsBlock when 'video' input is provided\n"
            " - Routes to PrepareLatentsBlock when no 'video' input is provided\n"
        )
