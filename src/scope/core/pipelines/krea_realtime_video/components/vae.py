# Modified from https://github.com/krea-ai/realtime-video
import os

import torch
import torch.nn as nn

from ..modules.vae import (
    CausalConv3d,
    WanVAE,
)
from ..modules.vae_block3 import VAEDecoder3d


class WanVAEWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Wan2.1-T2V-1.3B",
        model_dir: str | None = None,
        vae_path: str | None = None,
    ):
        super().__init__()

        # Determine paths with priority: specific paths > model_dir > default
        if vae_path is None:
            model_dir = model_dir if model_dir is not None else "wan_models"
            model_path = os.path.join(model_dir, model_name)
            vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")

        vae = WanVAE(vae_pth=vae_path)

        self.encoder = VAEEncoderWrapper(vae)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.decoder = VAEDecoderWrapper()
        state_dict = torch.load(vae_path, map_location="cpu")
        decoder_state_dict = {}
        for key, value in state_dict.items():
            if "decoder." in key or "conv2" in key:
                decoder_state_dict[key] = value
        self.decoder.load_state_dict(state_dict, strict=False)
        self.decoder.eval()
        self.decoder.requires_grad_(False)
        self.decoder_cache = [None] * 55

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        encoder_cache = [None] * 55
        output, _ = self.encoder(pixel, encoder_cache)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        output, self.decoder_cache = self.decoder(latent, *self.decoder_cache)
        return output

    def clear_cache(self):
        self.decoder_cache = [None] * 55


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.model.encoder
        self.conv1 = vae.model.conv1
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
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32)
        )  # use buffers to make sure that these numbers get casted
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.z_dim = 16

    def forward(
        self, z: torch.Tensor, feat_cache: list[torch.Tensor], stream: bool = False
    ):
        _, dtype = z.device, z.dtype
        scale = [self.mean.to(dtype=dtype), 1.0 / self.std.to(dtype=dtype)]

        # cache
        t = z.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
        # range_iter = range(iter_) if
        offset = 1
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0 and feat_cache[0] is None:
                out = self.encoder(
                    z[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                slice_start = i - 1
                if stream:
                    offset = 0
                    slice_start = i

                out_ = self.encoder(
                    z[
                        :,
                        :,
                        offset + 4 * slice_start : offset + 4 * (slice_start + 1),
                        :,
                        :,
                    ],
                    feat_cache=feat_cache,
                    feat_idx=self._enc_conv_idx,
                )
                if i == 0 and stream:
                    out = out_
                else:
                    out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        return mu, feat_cache


class VAEDecoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = VAEDecoder3d()
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

        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32)
        )  # use buffers to make sure that these numbers get casted
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.z_dim = 16
        self.conv2 = CausalConv3d(self.z_dim, self.z_dim, 1)

    def forward(self, z: torch.Tensor, *feat_cache: list[torch.Tensor]):
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        z = z.permute(0, 2, 1, 3, 4)
        feat_cache = list(feat_cache)

        _, dtype = z.device, z.dtype
        scale = [self.mean.to(dtype=dtype), 1.0 / self.std.to(dtype=dtype)]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        # print("iter_", iter_)
        x = self.conv2(z)
        for i in range(iter_):
            if i == 0:
                out, feat_cache = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=feat_cache
                )
            else:
                out_, feat_cache = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=feat_cache
                )
                out = torch.cat([out, out_], 2)

        out = out.float().clamp_(-1, 1)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        out = out.permute(0, 2, 1, 3, 4)
        return out, feat_cache
