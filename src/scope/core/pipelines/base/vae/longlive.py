"""LongLive VAE wrapper for batch-based encoding/decoding."""

import os

import torch

from ...longlive.modules.vae import _video_vae
from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD


class LongLiveVAE(torch.nn.Module):
    """VAE wrapper for LongLive with batch-based processing.

    This VAE processes entire video batches at once, providing higher quality
    but requiring more memory than streaming approaches.
    """

    def __init__(self, model_dir: str | None = None):
        super().__init__()

        # Use provided model_dir or default to "wan_models"
        model_dir = model_dir if model_dir is not None else "wan_models"

        self.mean = torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
        self.std = torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)

        # Init model
        vae_path = os.path.join(model_dir, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """Encode video pixels to latents with batch processing.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]

        Returns:
            Latent tensor [batch, frames, channels, height, width]
        """
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        """Decode latents to video pixels with optional caching.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width]
            use_cache: Whether to use cached decoding for temporal consistency

        Returns:
            Video tensor [batch, frames, channels, height, width] in range [-1, 1]
        """
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(
                decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def clear_cache(self):
        """Clear decoder cache for next sequence."""
        self.model.clear_cache()
