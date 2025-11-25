"""StreamDiffusionV2 VAE wrapper with streaming-friendly encoding/decoding."""

import os

import torch

from ...streamdiffusionv2.modules.vae import _video_vae
from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD


class StreamDiffusionV2VAE(torch.nn.Module):
    """VAE wrapper for StreamDiffusionV2 with streaming-friendly encoding/decoding.

    This VAE is optimized for real-time video processing with minimal memory overhead.
    """

    def __init__(self, model_dir: str | None = None):
        super().__init__()

        model_dir = model_dir if model_dir is not None else "wan_models"
        self.mean = torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
        self.std = torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)

        self.model = (
            _video_vae(
                pretrained_path=os.path.join(
                    model_dir, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
                ),
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode video pixels to latents with streaming-friendly processing.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]

        Returns:
            Latent tensor [batch, frames, channels, height, width]
        """
        latent = self.model.stream_encode(pixel)
        return latent.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        """Decode latents to video pixels with streaming-friendly processing.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width]
            use_cache: Whether to use decoder cache (always True for streaming)

        Returns:
            Video tensor [batch, frames, channels, height, width] in range [-1, 1]
        """
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to("cuda")
        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]
        output = self.model.stream_decode(zs, scale).float().clamp_(-1, 1)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def clear_cache(self):
        """Clear decoder cache for next sequence."""
        self.model.first_batch = True


class StreamDiffusionV2VAEWithLongLiveScaling(StreamDiffusionV2VAE):
    """StreamDiffusionV2 VAE with LongLive-style latent scaling for cross-pipeline use.

    This subclass applies the same latent normalization as LongLiveVAE, making it
    suitable for use in the LongLive pipeline's video-to-video mode where streaming
    efficiency is needed but latent distributions must match LongLive expectations.

    Note: This implementation uses subclassing for scaling. A future enhancement could
    parameterize scaling behavior in the base StreamDiffusionV2VAE class to avoid
    the need for separate classes.
    """

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode video pixels with LongLive-style scaling applied.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]

        Returns:
            Latent tensor [batch, frames, channels, height, width] with scaling
        """
        latent = self.model.stream_encode(pixel)

        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if isinstance(scale[0], torch.Tensor):
            latent = (latent - scale[0].view(1, latent.shape[1], 1, 1, 1)) * scale[
                1
            ].view(1, latent.shape[1], 1, 1, 1)
        else:
            latent = (latent - scale[0]) * scale[1]

        return latent.permute(0, 2, 1, 3, 4)
