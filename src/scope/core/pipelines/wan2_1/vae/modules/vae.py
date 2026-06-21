# Modified from https://github.com/chenfengxu714/StreamdiffusionV2
# Pruning functionality adapted from https://github.com/ModelTC/LightX2V
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "CacheState",
    "WanVAE",
    "_video_vae",
]

CACHE_T = 2


def _compute_new_cache(x: torch.Tensor, old_buf: torch.Tensor) -> torch.Tensor:
    """Compute new cache from input x and old buffer. Compile-safe (no list ops)."""
    cache_x = x[:, :, -CACHE_T:, :, :].contiguous().clone()
    if cache_x.shape[2] < CACHE_T:
        cache_x = torch.cat([old_buf[:, :, -1:, :, :], cache_x], dim=2)
    return cache_x


class CacheState:
    """Pre-allocated, type-stable cache for the VAE encoder and decoder.

    Replaces the mutable list-of-None/string/tensor pattern with structured
    tensor buffers. Zero-initialized buffers produce identical results to None
    cache slots because CausalConv3d with a zero prefix is numerically
    equivalent to CausalConv3d with zero-padding.

    Buffers are lazily allocated on first use (outside torch.compile) and then
    reused across sequence resets. After the first forward pass populates all
    slots, reset() zeros buffers without deallocating, so torch.compile never
    sees a None-to-Tensor type transition.

    Supports two buffer types:
    - Conv cache (CACHE_T frames): for CausalConv3d layers via update_conv_cache
    - Downsample cache (1 frame): for Downsample3d layers via update_downsample_cache
    """

    __slots__ = ("buffers", "populated", "is_rep_phase", "num_slots")

    def __init__(self, num_slots: int, device: torch.device, dtype: torch.dtype):
        self.num_slots = num_slots
        self.buffers: list[torch.Tensor | None] = [None] * num_slots
        self.populated = torch.zeros(num_slots, dtype=torch.bool, device=device)
        self.is_rep_phase = torch.zeros(num_slots, dtype=torch.bool, device=device)

    @torch.compiler.disable
    def _alloc_buffer(self, idx: int, like: torch.Tensor) -> torch.Tensor:
        """Allocate a zero buffer for slot idx. Called once per slot, never recompiled."""
        buf = torch.zeros(
            like.shape[0],
            like.shape[1],
            CACHE_T,
            like.shape[3],
            like.shape[4],
            device=like.device,
            dtype=like.dtype,
        )
        self.buffers[idx] = buf
        return buf

    @torch.compiler.disable
    def update_conv_cache(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """Prepare cache for a CausalConv3d. Returns the OLD buffer for the conv.

        Computes new cache frames from x, stores them, and returns the previous
        buffer so the caller can pass it to the convolution.

        Excluded from torch.compile because dynamo specializes on idx when
        indexing self.buffers (a heterogeneous Python list), causing O(num_slots)
        recompilations. The cache ops are cheap (slice/clone/cat); the
        convolutions between them are what benefit from compilation.
        """
        buf = self.buffers[idx]
        if buf is None:
            buf = self._alloc_buffer(idx, x)
        cache_x = x[:, :, -CACHE_T:, :, :].contiguous().clone()
        if cache_x.shape[2] < CACHE_T:
            if self.populated[idx]:
                cache_x = torch.cat([buf[:, :, -1:, :, :], cache_x], dim=2)
            else:
                pad = torch.zeros(
                    cache_x.shape[0],
                    cache_x.shape[1],
                    CACHE_T - cache_x.shape[2],
                    cache_x.shape[3],
                    cache_x.shape[4],
                    device=cache_x.device,
                    dtype=cache_x.dtype,
                )
                cache_x = torch.cat([pad, cache_x], dim=2)
        self.buffers[idx] = cache_x
        self.populated[idx] = True
        return buf

    @torch.compiler.disable
    def _alloc_downsample_buffer(self, idx: int, like: torch.Tensor) -> torch.Tensor:
        """Allocate a 1-frame zero buffer for a downsample slot."""
        buf = torch.zeros(
            like.shape[0],
            like.shape[1],
            1,
            like.shape[3],
            like.shape[4],
            device=like.device,
            dtype=like.dtype,
        )
        self.buffers[idx] = buf
        return buf

    @torch.compiler.disable
    def update_downsample_cache(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """Prepare cache for a Downsample3d time_conv. Returns the OLD 1-frame buffer.

        On the first call (not populated), stores x's last frame and returns
        a zero buffer. The caller must check populated[idx] beforehand to know
        whether to skip time_conv (first call) or run it (subsequent calls).
        """
        buf = self.buffers[idx]
        if buf is None:
            buf = self._alloc_downsample_buffer(idx, x)
        old = buf.contiguous().clone()
        self.buffers[idx] = x[:, :, -1:, :, :].contiguous().clone()
        self.populated[idx] = True
        return old

    @property
    def is_allocated(self) -> bool:
        """True when all buffer slots have been allocated (no None entries)."""
        return all(b is not None for b in self.buffers)

    def get_bufs(self) -> tuple[torch.Tensor, ...]:
        """Extract all buffers as a tuple for passing to compiled encoder/decoder."""
        return tuple(self.buffers)

    @torch.compiler.disable
    def set_bufs(self, new_bufs: list[torch.Tensor]):
        """Update buffers from list returned by compiled encoder/decoder."""
        for i, buf in enumerate(new_bufs):
            self.buffers[i] = buf
        self.populated[:] = True

    def reset(self):
        """Reset for a new sequence. Zeros existing buffers, preserving allocations."""
        for buf in self.buffers:
            if buf is not None:
                buf.zero_()
        self.populated.zero_()
        self.is_rep_phase.zero_()


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

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
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


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

        # layers
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

    def forward(
        self, x, feat_cache=None, feat_idx=None, *, cache=None, cache_bufs=None
    ):
        b, c, t, h, w = x.size()
        new_bufs = []
        if self.mode == "upsample3d":
            if cache_bufs is not None:
                idx = self.time_conv._cache_idx
                old = cache_bufs[idx]
                new_bufs.append(_compute_new_cache(x, old))
                x = self.time_conv(x, old)

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
            elif cache is not None:
                idx = self.time_conv._cache_idx
                if not cache.populated[idx] and not cache.is_rep_phase[idx]:
                    # First call: skip temporal processing, mark for next time
                    cache.is_rep_phase[idx] = True
                else:
                    # Second+ call: always pass buffer (zeros = no cache equivalent)
                    buf = cache.update_conv_cache(idx, x)
                    x = self.time_conv(x, buf)
                    cache.is_rep_phase[idx] = False

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
            elif feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
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
            if cache_bufs is not None:
                idx = self.time_conv._cache_idx
                old = cache_bufs[idx]
                cache_x = x[:, :, -1:, :, :].contiguous().clone()
                combined = torch.cat([old, x], 2)
                # Skip time_conv when combined frames < kernel size (first-frame
                # case with zero-init buffer). Dynamo creates separate traces
                # for each shape branch and caches them.
                if combined.shape[2] >= 3:
                    x = self.time_conv(combined)
                new_bufs.append(cache_x)
            elif cache is not None:
                idx = self.time_conv._cache_idx
                if not cache.populated[idx]:
                    cache.update_downsample_cache(idx, x)
                else:
                    old = cache.update_downsample_cache(idx, x)
                    x = self.time_conv(torch.cat([old, x], 2))
            elif feat_cache is not None:
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
        if cache_bufs is not None:
            return x, new_bufs
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        # conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        # init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[: c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2 :, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
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

    def forward(
        self, x, feat_cache=None, feat_idx=None, *, cache=None, cache_bufs=None
    ):
        h = self.shortcut(x)
        if cache_bufs is not None:
            new_bufs = []
            for layer in self.residual:
                if isinstance(layer, CausalConv3d):
                    old = cache_bufs[layer._cache_idx]
                    new_bufs.append(_compute_new_cache(x, old))
                    x = layer(x, old)
                else:
                    x = layer(x)
            return x + h, new_bufs
        elif cache is not None:
            for layer in self.residual:
                if isinstance(layer, CausalConv3d):
                    buf = cache.update_conv_cache(layer._cache_idx, x)
                    x = layer(x, buf)
                else:
                    x = layer(x)
        elif feat_cache is not None:
            for layer in self.residual:
                if isinstance(layer, CausalConv3d):
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
        else:
            for layer in self.residual:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x)
            .reshape(b * t, 1, c * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions (apply pruning to reduce channel dimensions)
        dims = [dim * u for u in [1] + dim_mult]
        dims = [int(d * (1 - pruning_rate)) for d in dims]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

        self._assign_cache_indices()

    def _assign_cache_indices(self):
        """Walk encoder layers in forward-pass order and set _cache_idx on each cacheable layer."""
        idx = 0
        # conv1
        self.conv1._cache_idx = idx
        idx += 1
        # downsamples
        for layer in self.downsamples:
            if isinstance(layer, ResidualBlock):
                for sublayer in layer.residual:
                    if isinstance(sublayer, CausalConv3d):
                        sublayer._cache_idx = idx
                        idx += 1
            elif isinstance(layer, Resample) and layer.mode == "downsample3d":
                layer.time_conv._cache_idx = idx
                idx += 1
        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                for sublayer in layer.residual:
                    if isinstance(sublayer, CausalConv3d):
                        sublayer._cache_idx = idx
                        idx += 1
        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                layer._cache_idx = idx
                idx += 1
        self._num_cache_slots = idx

    def forward(self, x, feat_cache=None, feat_idx=[0], *, cache=None, cache_bufs=None):
        new_bufs = []

        # conv1
        if cache_bufs is not None:
            old = cache_bufs[self.conv1._cache_idx]
            new_bufs.append(_compute_new_cache(x, old))
            x = self.conv1(x, old)
        elif cache is not None:
            buf = cache.update_conv_cache(self.conv1._cache_idx, x)
            x = self.conv1(x, buf)
        elif feat_cache is not None:
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

        # downsamples
        for layer in self.downsamples:
            if isinstance(layer, (ResidualBlock, Resample)):
                if cache_bufs is not None:
                    x, layer_new = layer(x, cache_bufs=cache_bufs)
                    new_bufs.extend(layer_new)
                elif cache is not None:
                    x = layer(x, cache=cache)
                elif feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                if cache_bufs is not None:
                    x, layer_new = layer(x, cache_bufs=cache_bufs)
                    new_bufs.extend(layer_new)
                elif cache is not None:
                    x = layer(x, cache=cache)
                elif feat_cache is not None:
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                if cache_bufs is not None:
                    idx = layer._cache_idx
                    old = cache_bufs[idx]
                    new_bufs.append(_compute_new_cache(x, old))
                    x = layer(x, old)
                elif cache is not None:
                    buf = cache.update_conv_cache(layer._cache_idx, x)
                    x = layer(x, buf)
                elif feat_cache is not None:
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
            else:
                x = layer(x)

        if cache_bufs is not None:
            return x, new_bufs
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions (apply pruning to reduce channel dimensions)
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        dims = [int(d * (1 - pruning_rate)) for d in dims]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

        # Assign static cache indices to all decoder CausalConv3d layers
        self._assign_cache_indices()

    def _assign_cache_indices(self):
        """Walk decoder layers in forward-pass order and set _cache_idx on each CausalConv3d."""
        idx = 0
        # conv1
        self.conv1._cache_idx = idx
        idx += 1
        # middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                for sublayer in layer.residual:
                    if isinstance(sublayer, CausalConv3d):
                        sublayer._cache_idx = idx
                        idx += 1
        # upsamples
        for layer in self.upsamples:
            if isinstance(layer, ResidualBlock):
                for sublayer in layer.residual:
                    if isinstance(sublayer, CausalConv3d):
                        sublayer._cache_idx = idx
                        idx += 1
            elif isinstance(layer, Resample) and layer.mode == "upsample3d":
                layer.time_conv._cache_idx = idx
                idx += 1
        # head
        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                layer._cache_idx = idx
                idx += 1
        self._num_cache_slots = idx

    def forward(self, x, cache=None, *, cache_bufs=None):
        new_bufs = []

        if cache_bufs is not None:
            old = cache_bufs[self.conv1._cache_idx]
            new_bufs.append(_compute_new_cache(x, old))
            x = self.conv1(x, old)
        elif cache is not None:
            buf = cache.update_conv_cache(self.conv1._cache_idx, x)
            x = self.conv1(x, buf)
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                if cache_bufs is not None:
                    x, layer_new = layer(x, cache_bufs=cache_bufs)
                    new_bufs.extend(layer_new)
                else:
                    x = layer(x, cache=cache)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, (ResidualBlock, Resample)):
                if cache_bufs is not None:
                    x, layer_new = layer(x, cache_bufs=cache_bufs)
                    new_bufs.extend(layer_new)
                else:
                    x = layer(x, cache=cache)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d):
                if cache_bufs is not None:
                    idx = layer._cache_idx
                    old = cache_bufs[idx]
                    new_bufs.append(_compute_new_cache(x, old))
                    x = layer(x, old)
                elif cache is not None:
                    buf = cache.update_conv_cache(layer._cache_idx, x)
                    x = layer(x, buf)
                else:
                    x = layer(x)
            else:
                x = layer(x)

        if cache_bufs is not None:
            return x, new_bufs
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        pruning_rate=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
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
        self._decoder_cache = None
        self._encoder_cache = None

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        # cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        # 对encode输入的x，按时间拆分为1、4、4、4....
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
        self.clear_cache()
        return mu

    def stream_encode(self, x):
        t = x.shape[2]
        if self.first_batch:
            if self._encoder_cache is not None:
                self._encoder_cache.reset()
            else:
                self._encoder_cache = CacheState(
                    self.encoder._num_cache_slots, x.device, x.dtype
                )

            # First batch always uses eager cache= path. Variable frame
            # counts (1 + t-1) would cause recompilation under torch.compile.
            # Only steady-state (4-frame chunks) is compiled. Bypass
            # torch.compile via _orig_mod when the encoder is compiled.
            enc = getattr(self.encoder, "_orig_mod", self.encoder)
            out = enc(x[:, :, :1, :, :], cache=self._encoder_cache)
            out_ = enc(x[:, :, 1:, :, :], cache=self._encoder_cache)
            out = torch.cat([out, out_], 2)
        else:
            cache = self._encoder_cache
            out = []
            for i in range(t // 4):
                if cache.is_allocated:
                    bufs = cache.get_bufs()
                    result, new_bufs = self.encoder(
                        x[:, :, i * 4 : (i + 1) * 4, :, :], cache_bufs=bufs
                    )
                    cache.set_bufs(new_bufs)
                    out.append(result)
                else:
                    out.append(
                        self.encoder(
                            x[:, :, i * 4 : (i + 1) * 4, :, :],
                            cache=cache,
                        )
                    )
            out = torch.cat(out, 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        return mu

    def decode(self, z, scale):
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        cache = CacheState(self.decoder._num_cache_slots, x.device, x.dtype)
        for i in range(iter_):
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], cache=cache)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], cache=cache)
                out = torch.cat([out, out_], 2)
        return out

    def stream_decode(self, z, scale):
        # z: [b,c,t,h,w]
        t = z.shape[2]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1
            )
        else:
            z = z / scale[1] + scale[0]
        x = self.conv2(z)
        if self.first_batch:
            if self._decoder_cache is not None:
                # Reuse pre-allocated buffers from a previous sequence
                self._decoder_cache.reset()
            else:
                self._decoder_cache = CacheState(
                    self.decoder._num_cache_slots, x.device, x.dtype
                )
            self.first_batch = False
            if self._decoder_cache.is_allocated:
                # Cache buffers already allocated (e.g. by compile_decoder warmup).
                # Use cache_bufs path to avoid graph breaks under torch.compile.
                out = []
                cache = self._decoder_cache
                for i in range(t):
                    bufs = cache.get_bufs()
                    result, new_bufs = self.decoder(
                        x[:, :, i : (i + 1), :, :], cache_bufs=bufs
                    )
                    cache.set_bufs(new_bufs)
                    out.append(result)
                out = torch.cat(out, 2)
            else:
                out = self.decoder(x[:, :, :1, :, :], cache=self._decoder_cache)
                out_ = self.decoder(x[:, :, 1:, :, :], cache=self._decoder_cache)
                out = torch.cat([out, out_], 2)
        else:
            out = []
            cache = self._decoder_cache
            for i in range(t):
                bufs = cache.get_bufs()
                result, new_bufs = self.decoder(
                    x[:, :, i : (i + 1), :, :], cache_bufs=bufs
                )
                cache.set_bufs(new_bufs)
                out.append(result)
            out = torch.cat(out, 2)
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self.clear_cache_decode()
        self.clear_cache_encode()

    def clear_cache_decode(self):
        # Keep the cache object (and its pre-allocated buffers) alive.
        # stream_decode will call reset() on next first_batch to zero
        # the buffers without deallocating.
        pass

    def clear_cache_encode(self):
        # Legacy feat_cache for encode() and _encode_with_cache()
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
        # CacheState for stream_encode is reset in stream_encode itself
        # when first_batch is True, so no action needed here.


def _video_vae(
    pretrained_path=None, z_dim=None, device="cpu", pruning_rate=0.0, **kwargs
):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.

    Args:
        pretrained_path: Path to checkpoint file (.pth or .safetensors)
        z_dim: Latent dimension
        device: Target device
        pruning_rate: Channel pruning rate (0.0 = full VAE, 0.75 = LightVAE)
        **kwargs: Additional model configuration
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        pruning_rate=pruning_rate,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    # load checkpoint (supports both .pth and .safetensors)
    logging.info(f"_video_vae: loading {pretrained_path}")
    if pretrained_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(pretrained_path, device=str(device))
    else:
        state_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state_dict, assign=True)

    return model


class WanVAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth="cache/vae_step_411000.pth",
        dtype=torch.float,
        device="cuda",
    ):
        self.dtype = dtype
        self.device = device

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
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                z_dim=z_dim,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0), self.scale)
                .float()
                .clamp_(-1, 1)
                .squeeze(0)
                for u in zs
            ]
