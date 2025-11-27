"""AutoPipelineBlocks for latent preparation routing.

This module provides block-level routing between text-to-video and video-to-video
latent preparation workflows. All pipelines can use this shared routing block.
"""

from diffusers.modular_pipelines import AutoPipelineBlocks, SequentialPipelineBlocks

from .noise_scale_controller import NoiseScaleControllerBlock
from .prepare_latents import PrepareLatentsBlock
from .prepare_video_latents import PrepareVideoLatentsBlock
from .preprocess_video import PreprocessVideoBlock


class TextToVideoLatentWorkflow(SequentialPipelineBlocks):
    """Text-to-video latent preparation workflow.

    Generates pure noise latents without any video preprocessing.
    """

    block_classes = [
        PrepareLatentsBlock,
    ]

    block_names = [
        "prepare_latents",
    ]


class VideoToVideoLatentWorkflow(SequentialPipelineBlocks):
    """Video-to-video latent preparation workflow.

    Preprocesses input video, applies motion-aware noise control,
    and encodes video to noisy latents.
    """

    block_classes = [
        PreprocessVideoBlock,
        NoiseScaleControllerBlock,
        PrepareVideoLatentsBlock,
    ]

    block_names = [
        "preprocess_video",
        "noise_scale_controller",
        "prepare_video_latents",
    ]


class AutoPrepareLatentsBlock(AutoPipelineBlocks):
    """Auto-routing block for latent preparation.

    Automatically routes between text-to-video and video-to-video
    latent preparation workflows based on whether 'video' input is provided.

    This block is shared across all pipelines (LongLive, StreamDiffusionV2,
    KreaRealtimeVideo) and provides a single point of routing for latent
    preparation logic.
    """

    block_classes = [
        VideoToVideoLatentWorkflow,
        TextToVideoLatentWorkflow,
    ]

    block_names = [
        "video_to_video_latents",
        "text_to_video_latents",
    ]

    block_trigger_inputs = [
        "video",
        None,
    ]

    @property
    def description(self):
        return (
            "AutoPrepareLatentsBlock: Auto-routing latent preparation block:\n"
            " - Routes to video-to-video workflow when 'video' input is provided\n"
            " - Routes to text-to-video workflow when no 'video' input is provided\n"
        )
