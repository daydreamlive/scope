import os

import torch

from ...streamdiffusionv2.modules.vae import _video_vae


class KreaStreamingVAE(torch.nn.Module):
    def __init__(
        self,
        model_name: str | None = None,
        model_dir: str | None = None,
        vae_path: str | None = None,
    ):
        super().__init__()

        # Determine VAE path with priority: vae_path > model_dir/model_name > default
        if vae_path is None:
            model_dir = model_dir if model_dir is not None else "wan_models"
            if model_name is None:
                model_name = "Wan2.1-T2V-1.3B"
            vae_path = os.path.join(model_dir, model_name, "Wan2.1_VAE.pth")

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

        # Initialize the underlying video VAE. We reuse the same Wan2.1 VAE
        # weights as StreamDiffusionV2 and LongLive for compatibility with the
        # Wan backbone, but expose Krea-specific streaming behaviour at the
        # wrapper level.
        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=16,
            )
            .eval()
            .requires_grad_(False)
        )

    # Streaming-friendly encode that is robust to very short sequences.
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # The Wan diffusion backbone and downstream blocks expect latents in
        # [batch, frames, channels, height, width] format. For typical Krea
        # streaming workflows we mirror the behaviour of StreamDiffusionV2 and
        # LongLive by:
        #   - Using stream_encode for multi-frame inputs (T >= 4) and applying
        #     LongLive-style whitening to match the latent distribution
        #     expected by the Wan backbone.
        #   - Falling back to the non-streaming encode path for very short
        #     sequences (T < 4), which are common during KV-cache
        #     recomputation, to avoid the torch.cat(empty) failure in
        #     stream_encode while preserving continuous latent geometry.
        device, dtype = pixel.device, pixel.dtype
        num_frames = pixel.shape[2]
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if num_frames < 4:
            # encode applies whitening using the provided scale internally.
            latent = self.model.encode(pixel, scale)
        else:
            latent = self.model.stream_encode(pixel)

            # Apply LongLive-style whitening so that latents match the
            # distribution expected by the Wan backbone and Krea's KV/cache
            # logic.
            if isinstance(scale[0], torch.Tensor):
                latent = (latent - scale[0].view(1, latent.shape[1], 1, 1, 1)) * scale[
                    1
                ].view(1, latent.shape[1], 1, 1, 1)
            else:
                latent = (latent - scale[0]) * scale[1]

        # Convert to [batch, frames, channels, height, width] for downstream
        # blocks (PrepareVideoLatentsBlock, DenoiseBlock, etc.).
        return latent.permute(0, 2, 1, 3, 4)

    # Streaming-friendly decode mirroring StreamDiffusionV2VAE.
    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        # Expect latents in [batch, frames, channels, height, width] and
        # convert to [batch, channels, frames, height, width] for the VAE.
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
        # We do not call self.model.clear_cache() here and instead we set the
        # first_batch flag, matching StreamDiffusionV2's streaming semantics.
        self.model.first_batch = True
