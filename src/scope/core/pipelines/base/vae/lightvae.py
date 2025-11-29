# Adopted from LightX2V: https://github.com/ModelTC/LightX2V
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open

from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD


class CausalConv3d(nn.Conv3d):
    """Causal 3d convolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        CACHE_T = 2
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = torch.zeros(
                        b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype
                    )
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2:
                        padding = torch.where(
                            feat_cache[idx][:, :, -1:, :, :] == 0,
                            torch.zeros_like(cache_x),
                            cache_x,
                        )
                        cache_x = torch.cat([padding, cache_x], dim=2)

                    x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        CACHE_T = 2
        if feat_idx is None:
            feat_idx = [0]
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )

                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Causal self-attention with a single head."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_downsample=None,
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        dims = [int(d * (1 - pruning_rate)) for d in dims]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        CACHE_T = 2
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_upsample=None,
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_upsample is None:
            temperal_upsample = [False, True, True]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        dims = [int(d * (1 - pruning_rate)) for d in dims]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=None):
        CACHE_T = 2

        if feat_idx is None:
            feat_idx = [0]

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]

                b, c, t, h, w = x.shape
                cache_x = torch.zeros(
                    b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype
                )
                fill_cache_x = x[:, :, -CACHE_T:, :, :].clone()
                cache_x[:, :, -fill_cache_x.shape[2] :, :, :] = fill_cache_x

                if fill_cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            fill_cache_x,
                        ],
                        dim=2,
                    )

                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        if feat_cache is not None:
            return x, feat_cache
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class LightVAE_(nn.Module):
    """LightVAE base model with pruning support."""

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=None,
        num_res_blocks=2,
        attn_scales=None,
        temperal_downsample=None,
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temperal_downsample is None:
            temperal_downsample = [True, True, False]
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
            pruning_rate,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            pruning_rate,
        )

        self.first_batch = True

    def encode(self, x, scale):
        self.clear_cache_encode()
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache_encode()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            result = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )

            if isinstance(result, tuple):
                out_frame, self._feat_map = result
            else:
                out_frame = result

            if i == 0:
                out = out_frame
            else:
                out = torch.cat([out, out_frame], 2)
        self.clear_cache()
        return out

    def cached_decode(self, z, scale):
        """Decode with cache (for streaming decode)."""
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        outs = []

        for i in range(iter_):
            self._conv_idx = [0]
            result = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )

            if isinstance(result, tuple):
                out, self._feat_map = result
            else:
                out = result

            outs.append(out)
        result = torch.cat(outs, 2)
        return result

    def stream_encode(self, x):
        """
        stream_encode: Streaming encode optimized for StreamDiffusionV2.

        Similar to WanVAE's stream_encode, handles first_batch initialization
        and processes frames in streaming-optimized chunks.

        Note: Returns raw mu without normalization, matching WanVAE behavior.

        Args:
            x: [B, C, T, H, W] pixel tensor

        Returns:
            [B, Cz, T, H/8, W/8] latent tensor (unnormalized)
        """
        t = x.shape[2]
        if self.first_batch:
            self.clear_cache_encode()
            self._enc_conv_idx = [0]
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            self._enc_conv_idx = [0]
            out_ = self.encoder(
                x[:, :, 1:, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
        else:
            out = []
            for i in range(t // 4):
                self._enc_conv_idx = [0]
                out.append(
                    self.encoder(
                        x[:, :, i * 4 : (i + 1) * 4, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                )
            out = torch.cat(out, 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def stream_decode(self, z, scale):
        """
        stream_decode: Streaming decode optimized for StreamDiffusionV2.

        Similar to WanVAE's stream_decode, handles first_batch initialization
        and processes frames with streaming cache management.

        Args:
            z: [B, Cz, T, H/8, W/8] latent tensor
            scale: [mean, 1/std] normalization parameters

        Returns:
            [B, C, T, H, W] pixel tensor
        """
        t = z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        if self.first_batch:
            self.clear_cache_decode()
            self.first_batch = False

            result = self.decoder(
                x[:, :, :1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            if isinstance(result, tuple):
                out, self._feat_map = result
            else:
                out = result

            result = self.decoder(
                x[:, :, 1:, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            if isinstance(result, tuple):
                out_, self._feat_map = result
            else:
                out_ = result

            out = torch.cat([out, out_], 2)
        else:
            out = []
            for i in range(t):
                self._conv_idx = [0]
                result = self.decoder(
                    x[:, :, i : (i + 1), :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                if isinstance(result, tuple):
                    out_frame, self._feat_map = result
                else:
                    out_frame = result
                out.append(out_frame)
            out = torch.cat(out, 2)
        return out

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_cache_encode(self):
        """clear_cache_encode: Clear encoder cache only."""
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def clear_cache_decode(self):
        """clear_cache_decode: Clear decoder cache only."""
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num


def _load_lightvae_weights(pretrained_path: str) -> dict:
    """Load LightVAE weights from .pth or .safetensors file."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"LightVAE checkpoint not found at: {pretrained_path}")

    if pretrained_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(pretrained_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    elif pretrained_path.endswith(".pth") or pretrained_path.endswith(".pt"):
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(
            f"Unsupported checkpoint format. Expected .safetensors, .pth, or .pt, got: {pretrained_path}"
        )

    return state_dict


class LightVAEWrapper(torch.nn.Module):
    """
    LightVAE wrapper for LongLive pipeline.

    This wrapper loads the LightVAE model (75% pruned WanVAE) and provides the same interface
    as WanVAEWrapper with:
    - encode_to_latent(pixel: [B,C,T,H,W]) -> [B,T,Cz,H/8,W/8]
    - decode_to_pixel(latent: [B,T,Cz,H/8,W/8], use_cache: bool=False) -> [B,T,C,H,W]

    Uses the same normalization as WanVAE.
    """

    def __init__(self, vae_path: str | None = None, model_dir: str | None = None):
        super().__init__()

        if vae_path is None:
            if model_dir is None:
                from scope.core.config import get_models_dir

                model_dir = str(get_models_dir())
            vae_path = os.path.join(model_dir, "Wan2.1-T2V-1.3B", "lightvaew2_1.pth")

        if not os.path.exists(vae_path):
            raise FileNotFoundError(
                f"LightVAE checkpoint not found at: {vae_path}\n"
                f"Please download lightvaew2_1.pth from lightx2v/Autoencoders HuggingFace repository."
            )

        self.mean = torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
        self.std = torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)

        cfg = {
            "dim": 96,
            "z_dim": 16,
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0,
            "pruning_rate": 0.75,
        }

        with torch.device("meta"):
            self.model = LightVAE_(**cfg)

        weights_dict = _load_lightvae_weights(vae_path)

        missing_keys = []
        unexpected_keys = []
        model_keys = set(dict(self.model.named_parameters()).keys()) | set(
            dict(self.model.named_buffers()).keys()
        )
        checkpoint_keys = set(weights_dict.keys())

        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys

        if missing_keys or unexpected_keys:
            error_msg = f"LightVAEWrapper: checkpoint mismatch for {vae_path}\n"
            if missing_keys:
                error_msg += f"  Missing keys in checkpoint ({len(missing_keys)}): {sorted(missing_keys)[:10]}...\n"
            if unexpected_keys:
                error_msg += f"  Unexpected keys in checkpoint ({len(unexpected_keys)}): {sorted(unexpected_keys)[:10]}...\n"
            error_msg += "\nThis likely means the checkpoint is not a valid LightVAE (lightvaew2_1.pth) checkpoint.\n"
            error_msg += (
                "Ensure you are using a LightVAE checkpoint with 75% pruning applied."
            )
            raise RuntimeError(error_msg)

        self.model.load_state_dict(weights_dict, assign=True)
        self.model.eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel space to latent space.

        Args:
            pixel: [batch_size, num_channels, num_frames, height, width]

        Returns:
            latent: [batch_size, num_frames, num_channels, height/8, width/8]
        """
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [self.model.encode(u.unsqueeze(0), scale).squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        """
        Decode latent space to pixel space.

        Args:
            latent: [batch_size, num_frames, num_channels, height, width]
            use_cache: if True, use cached decoding (batch size must be 1)

        Returns:
            pixel: [batch_size, num_frames, num_channels, height*8, width*8]
        """
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert (
                latent.shape[0] == 1
            ), "LightVAEWrapper.decode_to_pixel: Batch size must be 1 when using cache"

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
        for _batch_idx, u in enumerate(zs):
            output.append(
                decode_function(u.unsqueeze(0), scale).clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def clear_cache(self):
        """Clear cached state for streaming decode."""
        self.model.clear_cache()


class StreamingLightVAEWrapper(LightVAEWrapper):
    """Streaming-enabled LightVAE wrapper based on StreamDiffusionV2 streaming patterns.

    This subclass of LightVAEWrapper uses stream_encode and stream_decode methods
    for optimized real-time video processing with minimal memory overhead, similar
    to StreamDiffusionV2VAE but using the LightVAE model (75% pruned WanVAE).

    The streaming methods handle first_batch initialization and cache management
    automatically for efficient frame-by-frame processing.
    """

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


class StreamingLightVAEWrapperWithLongLiveScaling(StreamingLightVAEWrapper):
    """StreamingLightVAE wrapper with LongLive-style latent scaling for cross-pipeline use.

    This subclass applies the same latent normalization as LongLiveVAE, making it
    suitable for use in the LongLive pipeline's video-to-video mode where streaming
    efficiency is needed but latent distributions must match LongLive expectations.

    Note: This implementation uses subclassing for scaling. A future enhancement could
    parameterize scaling behavior in the base StreamingLightVAEWrapper class to avoid
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
