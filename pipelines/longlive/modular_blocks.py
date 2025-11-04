import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)
from torch import Tensor


class VideoEncodeBlock(ModularPipelineBlocks):
    """Generic VAE encoding block - works with ANY VAEInterface implementation.

    This block encodes pixel frames to latents. It can work with:
    - LongLive VAE (via LongLiveVAEAdapter)
    - StreamDiffusion VAE (via different adapter)
    - Krea VAE (via different adapter)
    - Any other VAE that implements VAEInterface

    The block doesn't care about the concrete implementation.
    """

    model_name = "Generic"

    @property
    def description(self) -> str:
        return "Encode pixel frames to latent space using VAEInterface"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return []

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "pixels", type_hint=Tensor, description="Pixel frames [B, C, T, H, W]"
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=Tensor,
                description="Encoded latent frames [B, T, C_latent, H//8, W//8]",
            ),
        ]

    def __call__(self, components, state: PipelineState):
        """Encode pixels to latents using VAEInterface.

        This method depends ONLY on the VAEInterface protocol.
        It calls components.vae.encode_to_latent() which is defined by the protocol.
        """
        block_state = self.get_block_state(state)

        pixels = block_state.pixels
        if pixels is None:
            raise ValueError("VideoEncodeBlock: 'pixels' input is required")

        # This is the key line: we call the protocol method
        # components.vae could be ANY adapter that implements VAEInterface
        latents = components.vae.encode_to_latent(pixels)

        # Update state with encoded latents
        block_state.latents = latents
        self.set_block_state(state, block_state)

        return components, state


class VideoDecodeBlock(ModularPipelineBlocks):
    """Generic VAE decoding block - works with ANY VAEInterface implementation.

    This block decodes latent frames to pixels. Like VideoEncodeBlock,
    it works with any VAE adapter implementing VAEInterface.
    """

    model_name = "Generic"

    @property
    def description(self) -> str:
        return "Decode latent frames to pixel space using VAEInterface"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return []

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                type_hint=Tensor,
                description="Latent frames [B, T, C_latent, H//8, W//8]",
            ),
            InputParam(
                "use_cache",
                type_hint=bool,
                description="Whether to use temporal cache for streaming",
                default=False,
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "pixels",
                type_hint=Tensor,
                description="Decoded pixel frames [B, T, C, H, W]",
            ),
        ]

    def __call__(self, components, state: PipelineState):
        """Decode latents to pixels using VAEInterface.

        This method depends ONLY on the VAEInterface protocol.
        It calls components.vae.decode_to_pixel() which is defined by the protocol.
        """
        block_state = self.get_block_state(state)

        latents = block_state.latents
        use_cache = getattr(block_state, "use_cache", False)

        if latents is None:
            raise ValueError("VideoDecodeBlock: 'latents' input is required")

        # This is the key line: we call the protocol method
        # components.vae could be ANY adapter that implements VAEInterface
        pixels = components.vae.decode_to_pixel(latents, use_cache=use_cache)

        # Update state with decoded pixels
        block_state.pixels = pixels
        self.set_block_state(state, block_state)

        return components, state


class VAECacheClearBlock(ModularPipelineBlocks):
    """Generic VAE cache clearing block - works with ANY VAEInterface implementation."""

    model_name = "Generic"

    @property
    def description(self) -> str:
        return "Clear VAE internal caches using VAEInterface"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return []

    @property
    def inputs(self) -> list[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    def __call__(self, components, state: PipelineState):
        """Clear VAE caches using VAEInterface.

        This method depends ONLY on the VAEInterface protocol.
        """
        # This is the key line: we call the protocol method
        components.vae.clear_cache()

        return components, state
