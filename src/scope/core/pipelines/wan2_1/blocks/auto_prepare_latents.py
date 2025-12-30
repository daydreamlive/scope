"""AutoPipelineBlocks for video preprocessing and latent preparation routing.

This module provides block-level routing between text-to-video and video-to-video
workflows at two critical points:
1. Video preprocessing (after SetupCachesBlock)
2. Latent preparation (after SetupCachesBlock)

Video preprocessing runs after SetupCachesBlock to ensure that when caches are reset
(e.g., when prompts change), the preprocessing uses the correct current_start_frame value.
"""

from typing import Any

from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)

from .noise_scale_controller import NoiseScaleControllerBlock
from .prepare_latents import PrepareLatentsBlock
from .prepare_video_latents import PrepareVideoLatentsBlock
from .preprocess_video import PreprocessVideoBlock


class NoOpBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def description(self) -> str:
        return "NoOpBlock is a block that does nothing"

    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        return components, state


class V2VPreprocessingWorkflow(SequentialPipelineBlocks):
    """Video preprocessing workflow for V2V mode.

    Preprocesses input video and applies motion-aware noise control.
    Runs AFTER SetupCachesBlock to ensure correct frame count when caches are reset.
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

    This runs AFTER SetupCachesBlock to ensure correct frame count when caches are reset.
    """

    block_classes = [
        V2VPreprocessingWorkflow,
        PreprocessVideoBlock,
        NoOpBlock,
    ]

    block_names = [
        "v2v_preprocessing",
        "preprocess_video",
        "noop",
    ]

    block_trigger_inputs = ["video", "vace_input_frames", None]

    @property
    def description(self):
        return (
            "AutoPreprocessVideoBlock: Routes video preprocessing after cache setup:\n"
            " - Routes to V2VPreprocessingWorkflow when 'video' input is provided\n"
            " - Routes to PreprocessVideoBlock when 'vace_input_frames' input is provided\n"
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
