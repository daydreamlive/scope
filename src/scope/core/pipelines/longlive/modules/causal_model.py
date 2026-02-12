# Modified from https://github.com/NVlabs/LongLive
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from scope.core.pipelines.wan2_1.modules.attention import attention
from .model import (
    WAN_CROSSATTENTION_CLASSES,
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)


def precompute_freqs_i(freqs, f, h, w, start_frame=0):
    """Precompute RoPE frequency tensor outside the compiled region.

    Args:
        freqs: Base frequency table, shape [max_seq_len, head_dim // 2]
        f: Number of temporal frames
        h: Spatial height (after patching)
        w: Spatial width (after patching)
        start_frame: Starting frame index for temporal position encoding

    Returns:
        freqs_i: Precomputed frequency tensor, shape [f*h*w, 1, head_dim // 2]
    """
    c = freqs.shape[1]
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    freqs_i = torch.cat(
        [
            freqs_split[0][start_frame : start_frame + f]
            .view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)

    return freqs_i


def causal_rope_apply_precomputed(x, freqs_i):
    """Apply precomputed RoPE frequencies. No grid_sizes, no start_frame, no .item() calls.

    Args:
        x: Input tensor, shape [B, seq_len, num_heads, head_dim]
        freqs_i: Precomputed frequency tensor, shape [seq_len, 1, head_dim // 2]

    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    n, c = x.size(2), x.size(3) // 2
    seq_len = x.shape[1]

    x_rope = torch.view_as_complex(
        x.to(torch.float64).reshape(-1, seq_len, n, c, 2)
    )
    x_rope = torch.view_as_real(x_rope * freqs_i).flatten(3)

    return x_rope.type_as(x)


class CausalWanSelfAttention(nn.Module):
    def __init__(
        self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.frame_seq_length = 0  # Set by SetupCachesBlock at runtime

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs_i, kv_cache):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] (pre-norm'd and modulated input)
            freqs_i(Tensor): Precomputed RoPE frequencies, shape [L, 1, head_dim // 2]
            kv_cache(dict): Cache dict with 'k' and 'v' tensors
        Returns:
            Tuple of (output, (new_roped_k, new_v))
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        roped_q = causal_rope_apply_precomputed(q, freqs_i).type_as(v)
        roped_k = causal_rope_apply_precomputed(k, freqs_i).type_as(v)

        # Concatenate cache context with new KVs: 9 + 3 = 12 frames
        attn_k = torch.cat([kv_cache["k"], roped_k], dim=1)
        attn_v = torch.cat([kv_cache["v"], v], dim=1)

        x = attention(roped_q, attn_k, attn_v)

        # output
        x = x.flatten(2)
        x = self.o(x)

        return x, (roped_k, v)


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, eps
        )
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, freqs_i, context, kv_cache):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs_i(Tensor): Precomputed RoPE frequencies
            context(Tensor): Text embeddings
            kv_cache(dict): Cache dict with 'k' and 'v' tensors
        Returns:
            Tuple of (output, (new_roped_k, new_v))
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y, new_kv = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            freqs_i,
            kv_cache,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # cross-attention
        x = x + self.cross_attn(self.norm3(x), context, None)

        # ffn
        y = self.ffn(
            (
                self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[4])
                + e[3]
            ).flatten(1, 2)
        )
        x = x + (
            y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]
        ).flatten(1, 2)

        return x, new_kv


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1])
            + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["WanAttentionBlock"]
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Set by SetupCachesBlock at runtime
        self.frame_seq_length = 0

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    local_attn_size,
                    sink_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def _roll_update_cache(self, kv_cache, new_kvs, current_start):
        """Roll the KV cache and insert new KVs.

        During the filling phase (sink not yet active), the entire cache is rolled.
        Once the cache is full (sink active), only the non-sink portion is rolled,
        preserving the first sink_size frames as attention anchors.
        """
        cache_tokens = kv_cache[0]["k"].shape[1]
        sink_tokens = self.sink_size * self.frame_seq_length
        sink_active = current_start >= cache_tokens
        num_new = new_kvs[0][0].shape[1]

        for i, (new_k, new_v) in enumerate(new_kvs):
            if sink_active:
                # Roll only the non-sink portion, keep sink pinned
                kv_cache[i]["k"][:, sink_tokens:] = torch.roll(
                    kv_cache[i]["k"][:, sink_tokens:], -num_new, dims=1
                )
                kv_cache[i]["v"][:, sink_tokens:] = torch.roll(
                    kv_cache[i]["v"][:, sink_tokens:], -num_new, dims=1
                )
            else:
                # Full roll during filling phase
                kv_cache[i]["k"] = torch.roll(kv_cache[i]["k"], -num_new, dims=1)
                kv_cache[i]["v"] = torch.roll(kv_cache[i]["v"], -num_new, dims=1)
            # Overwrite the wrapped-around junk at the end with new values
            kv_cache[i]["k"][:, -num_new:] = new_k
            kv_cache[i]["v"][:, -num_new:] = new_v

    def _forward_inference(
        self,
        x,
        t,
        context,
        kv_cache: list = None,
        current_start: int = 0,
        update_cache: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.

        Args:
            x (Tensor): Input video tensor with shape [B, C_in, F, H, W]
            t (Tensor): Diffusion timesteps tensor of shape [B, F]
            context (Tensor): Text embeddings tensor with shape [B, L, C]
            kv_cache (list): List of cache dicts (one per block), each with 'k' and 'v'
            current_start (int): Starting position in tokens
            update_cache (bool): Whether to update the cache after forward pass
        Returns:
            Tensor: Denoised video tensor with shape [B, C_out, F, H, W]
        """
        device = self.patch_embedding.weight.device
        self.freqs = self.freqs.to(device)

        # Patch embedding
        x = self.patch_embedding(x)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)

        # Precompute RoPE frequencies (outside compiled region)
        start_frame = current_start // (h * w)
        freqs_i = precompute_freqs_i(self.freqs, f, h, w, start_frame)

        # Time and text embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        if context.shape[1] < self.text_len:
            padded_context = torch.zeros(
                (context.shape[0], self.text_len, context.shape[2]),
                dtype=context.dtype,
                device=context.device,
            )
            padded_context[:, : context.shape[1], :] = context
            context = padded_context
        context = self.text_embedding(context)

        # Block loop -- uniform path for all modes
        new_kvs = []
        for block_index, block in enumerate(self.blocks):
            x, new_kv = block(
                x,
                e=e0,
                freqs_i=freqs_i,
                context=context,
                kv_cache=kv_cache[block_index],
            )
            new_kvs.append(new_kv)

        # Cache update (outside compiled region)
        if update_cache and kv_cache is not None:
            self._roll_update_cache(kv_cache, new_kvs, current_start)

        # Head and unpatchify
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, f, h, w)
        return x

    def forward(self, *args, **kwargs):
        return self._forward_inference(*args, **kwargs)

    def unpatchify(self, x, f, h, w):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor): Batch of patchified features, shape [B, F*H*W, C_out * prod(patch_size)]
            f, h, w (int): Grid dimensions (frames, height, width) after patching

        Returns:
            Tensor: Reconstructed video tensor with shape [B, C_out, F*pt, H*ph, W*pw]
        """
        c = self.out_dim
        x = x.view(x.shape[0], f, h, w, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(
            x.shape[0],
            c,
            f * self.patch_size[0],
            h * self.patch_size[1],
            w * self.patch_size[2],
        )
        return x
