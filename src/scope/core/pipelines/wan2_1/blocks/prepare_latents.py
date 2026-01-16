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


class PrepareLatentsBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return "Prepare Latents block that generates empty latents (noise) for video generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
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
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
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

        generator_param = next(components.generator.model.parameters())
        latent_height = (
            block_state.height // components.config.vae_spatial_downsample_factor
        )
        latent_width = (
            block_state.width // components.config.vae_spatial_downsample_factor
        )

        # The default param for InputParam does not work right now
        # The workaround is to set the default values here
        base_seed = block_state.base_seed
        if base_seed is None:
            base_seed = 42

        # Create generator from seed for reproducible generation
        block_seed = base_seed + block_state.current_start_frame
        rng = torch.Generator(device=generator_param.device).manual_seed(block_seed)

        # Determine number of latent frames to generate
        num_latent_frames = components.config.num_frame_per_block
        # VAE stream_decode requires at least 2 latent frames on first batch
        # because it splits into x[:,:,:1,:,:] and x[:,:,1:,:,:] (second part must be non-empty)
        if block_state.current_start_frame == 0 and num_latent_frames < 2:
            num_latent_frames = 2

        # Generate empty latents (noise)
        latents = torch.randn(
            [
                1,  # batch_size
                num_latent_frames,
                16,
                latent_height,
                latent_width,
            ],
            device=generator_param.device,
            dtype=generator_param.dtype,
            generator=rng,
        )

        block_state.latents = latents
        block_state.generator = rng

        self.set_block_state(state, block_state)
        return components, state
