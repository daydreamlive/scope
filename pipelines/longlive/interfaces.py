"""Protocol definitions for LongLive modular pipeline components.

These are abstract interfaces that define contracts for pipeline components.
Any implementation conforming to these protocols can be swapped in.
"""

from typing import Protocol

from torch import Tensor


class VAEInterface(Protocol):
    """Protocol: Standard interface for all VAE implementations.

    Any VAE adapter must implement these methods to be compatible with modular blocks.
    """

    def encode_to_latent(self, pixels: Tensor) -> Tensor:
        """Encode pixel-space frames to latent space.

        Args:
            pixels: [B, C, T, H, W] pixel frames

        Returns:
            latents: [B, T, C_latent, H//8, W//8] latent frames

        Guarantees:
            - Output dtype MUST match input dtype
            - Must not modify internal decode cache state
        """
        ...

    def decode_to_pixel(self, latents: Tensor, use_cache: bool = False) -> Tensor:
        """Decode latent frames to pixel space.

        Args:
            latents: [B, T, C_latent, H//8, W//8] latent frames
            use_cache: Whether to use temporal cache for streaming

        Returns:
            pixels: [B, T*temporal_upsample, C, H, W] pixel frames

        Guarantees:
            - Output dtype MUST match input dtype
            - If use_cache=True, maintains temporal coherence
        """
        ...

    def clear_cache(self) -> None:
        """Clear all internal caches (both encode and decode)"""
        ...

    @property
    def supports_streaming(self) -> bool:
        """Whether this VAE supports stateful streaming with temporal cache"""
        ...

    @property
    def temporal_upsample_factor(self) -> int:
        """How many pixel frames are generated per latent frame (e.g., 4x)"""
        ...

    @property
    def dtype_stable(self) -> bool:
        """Whether output dtype always matches input dtype"""
        ...
