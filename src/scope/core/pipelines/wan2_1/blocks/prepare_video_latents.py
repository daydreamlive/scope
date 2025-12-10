from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)
from diffusers.utils import logging as diffusers_logging

logger = diffusers_logging.get_logger(__name__)


class PrepareVideoLatentsBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_temporal_downsample_factor", 4),
        ]

    @property
    def description(self) -> str:
        return "Prepare Video Latents block that generates noisy latents for a video that will be used for video generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "video",
                required=True,
                type_hint=list[torch.Tensor] | torch.Tensor,
                description="Input video to convert into noisy latents",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                default=42,
                description="Base seed for random number generation",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Amount of noise added to video",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            OutputParam("generator", description="Random number generator"),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        target_num_frames = (
            components.config.num_frame_per_block
            * components.config.vae_temporal_downsample_factor
        )
        # The first block will require an additional frame due to behavior of VAE
        if block_state.current_start_frame == 0:
            target_num_frames += 1

        input_video = block_state.video
        _, _, num_frames, _, _ = input_video.shape
        # If we do not have enough frames use linear interpolation to resample existing frames
        # to meet the required number of frames
        if num_frames != target_num_frames:
            indices = (
                torch.linspace(
                    0,
                    num_frames - 1,
                    target_num_frames,
                    device=input_video.device,
                )
                .round()
                .long()
            )
            input_video = input_video[:, :, indices]

        # Encode frames to latents using VAE
        # VAE returns [B, F, C, H, W] which is what DenoiseBlock/Generator expect
        latents = components.vae.encode_to_latent(input_video)

        # The default param for InputParam does not work right now
        # The workaround is to set the default values here
        base_seed = block_state.base_seed
        if base_seed is None:
            base_seed = 42

        # PROOF HACK 1: Correct current_start_frame for seed derivation
        corrected_current_start_frame = block_state.current_start_frame
        if block_state.current_start_frame > 0:
            corrected_current_start_frame = block_state.current_start_frame - 1
            logger.info(
                f"PROOF HACK 1 (PrepareVideoLatentsBlock): Corrected current_start_frame from {block_state.current_start_frame} to {corrected_current_start_frame} for seed"
            )

        # Create generator from seed for reproducible generation
        block_seed = base_seed + corrected_current_start_frame
        logger.info(
            f"PrepareVideoLatentsBlock: current_start_frame={block_state.current_start_frame}, "
            f"corrected={corrected_current_start_frame}, base_seed={base_seed}, block_seed={block_seed}, target_num_frames={target_num_frames}"
        )
        rng = torch.Generator(device=components.config.device).manual_seed(block_seed)

        # Generate empty latents (noise)
        noise = torch.randn(
            latents.shape,
            device=components.config.device,
            dtype=components.config.dtype,
            generator=rng,
        )
        # Determine how noisy the latents should be
        # Higher noise scale -> noiser latents, less of inputs preserved
        # Lower noise scale -> less noisy latents, more of inputs preserved
        block_state.latents = noise * block_state.noise_scale + latents * (
            1 - block_state.noise_scale
        )
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
