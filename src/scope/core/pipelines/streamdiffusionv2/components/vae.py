# Modified from https://github.com/chenfengxu714/StreamDiffusionV2
import os

import torch

from ..modules.vae import _video_vae


class WanVAEWrapper(torch.nn.Module):
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

        # init model
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

    # Streaming friendly
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        return self.model.stream_encode(pixel)

    # Streaming friendly
    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
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
        # We do not call self.model.clear_cache() here
        # and instead we set the first_batch flag
        self.model.first_batch = True
