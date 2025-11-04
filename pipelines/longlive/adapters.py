"""Adapters that make LongLive components conform to protocol interfaces.

These adapters wrap concrete implementations (WanVAEWrapper, WanTextEncoder, etc.)
and ensure they conform to the abstract protocols defined in interfaces.py.
"""

from torch import Tensor


class LongLiveVAEAdapter:
    """Adapter: Makes WanVAEWrapper conform to VAEInterface protocol.

    WanVAEWrapper (from pipelines.base.wan2_1.wrapper) has the right methods
    but calls .float() internally. This adapter preserves dtype stability.
    """

    def __init__(self, wan_vae):
        """Args:
        wan_vae: Instance of WanVAEWrapper from pipelines.base.wan2_1.wrapper
        """
        self.vae = wan_vae

    def encode_to_latent(self, pixels: Tensor) -> Tensor:
        """Encode pixel-space frames to latent space.

        WanVAEWrapper.encode_to_latent:
        - Input: [B, C, T, H, W] pixel frames
        - Output: [B, T, C_latent, H//8, W//8] latent frames
        - Note: Calls .float() internally, so we convert back to maintain dtype stability
        """
        original_dtype = pixels.dtype
        result = self.vae.encode_to_latent(pixels)

        # WanVAEWrapper.encode_to_latent calls .float() internally (line 175)
        # Convert back to input dtype to maintain dtype stability guarantee
        if result.dtype != original_dtype:
            result = result.to(original_dtype)

        return result

    def decode_to_pixel(self, latents: Tensor, use_cache: bool = False) -> Tensor:
        """Decode latent frames to pixel space.

        WanVAEWrapper.decode_to_pixel:
        - Input: [B, T, C_latent, H//8, W//8] latent frames
        - Output: [B, T, C, H, W] pixel frames
        - Supports use_cache parameter for streaming via model.cached_decode
        """
        original_dtype = latents.dtype
        result = self.vae.decode_to_pixel(latents, use_cache=use_cache)

        # WanVAEWrapper.decode_to_pixel calls .float() internally (line 204)
        # Convert back to input dtype to maintain dtype stability guarantee
        if result.dtype != original_dtype:
            result = result.to(original_dtype)

        return result

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self.vae.clear_cache()

    @property
    def supports_streaming(self) -> bool:
        """WanVAE supports streaming via model.cached_decode."""
        return True

    @property
    def temporal_upsample_factor(self) -> int:
        """WanVAE outputs 1 pixel frame per latent frame (no temporal upsampling)."""
        return 1

    @property
    def dtype_stable(self) -> bool:
        """After our dtype conversion in adapter, it becomes dtype stable."""
        return True
