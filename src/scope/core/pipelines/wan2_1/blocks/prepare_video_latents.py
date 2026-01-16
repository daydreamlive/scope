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

        # Encode frames to latents using VAE
        # VAE returns [B, F, C, H, W] which is what DenoiseBlock/Generator expect
        latents = components.vae.encode_to_latent(block_state.video)

        # The default param for InputParam does not work right now
        # The workaround is to set the default values here
        base_seed = block_state.base_seed
        if base_seed is None:
            base_seed = 42

        # Create generator from seed for reproducible generation
        block_seed = base_seed + block_state.current_start_frame
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
