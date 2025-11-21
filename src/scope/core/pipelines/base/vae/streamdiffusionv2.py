"""StreamDiffusionV2 VAE wrapper with streaming-friendly encoding/decoding."""

import os

import torch

from ...streamdiffusionv2.modules.vae import _video_vae


class StreamDiffusionV2VAE(torch.nn.Module):
    """VAE wrapper for StreamDiffusionV2 with streaming-friendly encoding/decoding.

    This VAE is optimized for real-time video processing with minimal memory overhead.
    """

    def __init__(self, model_dir: str | None = None):
        super().__init__()

        model_dir = model_dir if model_dir is not None else "wan_models"
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

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
    the need for separate strategy classes.
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
