# Modified from https://github.com/NVlabs/LongLive
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .model import (
    WAN_CROSSATTENTION_CLASSES,
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)


def precompute_freqs_i(freqs, f, h, w, start_frame=0):
    """Precompute RoPE cos/sin tensors outside the compiled region.

    Args:
        freqs: Base frequency table (complex), shape [max_seq_len, head_dim // 2]
        f: Number of temporal frames
        h: Spatial height (after patching)
        w: Spatial width (after patching)
        start_frame: Starting frame index for temporal position encoding

    Returns:
        Tuple of (freqs_cos, freqs_sin), each shape [1, f*h*w, 1, head_dim // 2]
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

    # Convert complex frequencies to real cos/sin for compile-friendly RoPE.
    # Shape: [1, seq_len, 1, head_dim // 2] for broadcasting with [B, seq_len, heads, head_dim // 2]
    freqs_cos = freqs_i.real.float().unsqueeze(0)
    freqs_sin = freqs_i.imag.float().unsqueeze(0)

    return freqs_cos, freqs_sin


def causal_rope_apply_precomputed(x, freqs_cos, freqs_sin):
    """Apply precomputed RoPE using real-valued sin/cos. Fully torch.compile friendly.

    Args:
        x: Input tensor, shape [B, seq_len, num_heads, head_dim]
        freqs_cos: Cosine frequencies, shape [1, seq_len, 1, head_dim // 2]
        freqs_sin: Sine frequencies, shape [1, seq_len, 1, head_dim // 2]

    Returns:
        Tensor with rotary embeddings applied, same shape as x.
    """
    # Split into even/odd pairs and apply rotation in float32
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    # Complex rotation: (a + bi)(cos + i*sin) = (a*cos - b*sin) + (a*sin + b*cos)i
    rotated_even = x_even * freqs_cos - x_odd * freqs_sin
    rotated_odd = x_even * freqs_sin + x_odd * freqs_cos

    # Interleave back: [even0, odd0, even1, odd1, ...]
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2).type_as(x)


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

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, freqs_cos, freqs_sin, cache_k, cache_v, mask):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] (pre-norm'd and modulated input)
            freqs_cos(Tensor): RoPE cosines, shape [1, L, 1, head_dim // 2]
            freqs_sin(Tensor): RoPE sines, shape [1, L, 1, head_dim // 2]
            cache_k(Tensor): Full static cache keys, shape [B, CACHE_SIZE, num_heads, head_dim]
            cache_v(Tensor): Full static cache values, shape [B, CACHE_SIZE, num_heads, head_dim]
            mask(Tensor): Attention mask, shape [1, 1, 1, CACHE_SIZE + L], 0=attend, -inf=ignore
        Returns:
            Tuple of (output, (new_roped_k, new_v))
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        roped_q = causal_rope_apply_precomputed(q, freqs_cos, freqs_sin)
        roped_k = causal_rope_apply_precomputed(k, freqs_cos, freqs_sin)

        attn_k = torch.cat([cache_k, roped_k], dim=1)
        attn_v = torch.cat([cache_v, v], dim=1)

        x = torch.nn.functional.scaled_dot_product_attention(
            roped_q.transpose(1, 2),
            attn_k.transpose(1, 2),
            attn_v.transpose(1, 2),
            attn_mask=mask,
        ).transpose(1, 2)

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

    def forward(self, x, e, freqs_cos, freqs_sin, context, cache_k, cache_v, mask):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs_cos(Tensor): RoPE cosines
            freqs_sin(Tensor): RoPE sines
            context(Tensor): Text embeddings
            cache_k(Tensor): Full static cache keys
            cache_v(Tensor): Full static cache values
            mask(Tensor): Attention mask for SDPA, shape [1, 1, 1, CACHE_SIZE + L]
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
            freqs_cos,
            freqs_sin,
            cache_k,
            cache_v,
            mask,
        )

        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(
            1, 2
        )

        # cross-attention (inlined with graph-safe attention to avoid graph breaks)
        cx = self.norm3(x)
        b_ca, n_ca, d_ca = cx.size(0), self.self_attn.num_heads, self.self_attn.head_dim
        cq = self.cross_attn.norm_q(self.cross_attn.q(cx)).view(b_ca, -1, n_ca, d_ca)
        ck = self.cross_attn.norm_k(self.cross_attn.k(context)).view(b_ca, -1, n_ca, d_ca)
        cv = self.cross_attn.v(context).view(b_ca, -1, n_ca, d_ca)
        ca_out = torch.nn.functional.scaled_dot_product_attention(
            cq.transpose(1, 2), ck.transpose(1, 2), cv.transpose(1, 2),
        ).transpose(1, 2).flatten(2)
        x = x + self.cross_attn.o(ca_out)

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

    def forward(self, x, t, context, freqs_cos, freqs_sin, cache_ks, cache_vs, mask):
        r"""
        Pure tensor forward pass (ONNX-exportable).

        All preprocessing (RoPE precomputation, context padding, cache tensor
        extraction, mask construction) and postprocessing (cache updates) are
        handled by the caller (WanDiffusionWrapper in generator.py).

        Args:
            x (Tensor): Input video [B, C_in, F, H, W]
            t (Tensor): Diffusion timesteps [B, F]
            context (Tensor): Pre-padded text embeddings [B, text_len, text_dim]
            freqs_cos (Tensor): Precomputed RoPE cosines [1, seq_len, 1, head_dim // 2]
            freqs_sin (Tensor): Precomputed RoPE sines [1, seq_len, 1, head_dim // 2]
            cache_ks (Tensor): Stacked cache keys [num_layers, B, cache_size, num_heads, head_dim]
            cache_vs (Tensor): Stacked cache values [num_layers, B, cache_size, num_heads, head_dim]
            mask (Tensor): Attention mask [1, 1, 1, cache_size + seq_len]
        Returns:
            Tuple of:
                output (Tensor): Denoised video [B, C_out, F', H', W']
                new_ks (Tensor): New keys [num_layers, B, seq_len, num_heads, head_dim]
                new_vs (Tensor): New values [num_layers, B, seq_len, num_heads, head_dim]
        """
        x = self.patch_embedding(x)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )
        context = self.text_embedding(context)

        new_ks = []
        new_vs = []
        for i, block in enumerate(self.blocks):
            x, (new_k, new_v) = block(
                x,
                e=e0,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                context=context,
                cache_k=cache_ks[i],
                cache_v=cache_vs[i],
                mask=mask,
            )
            new_ks.append(new_k)
            new_vs.append(new_v)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # x = self.unpatchify(x, f, h, w)
        return x, torch.stack(new_ks), torch.stack(new_vs)

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
        # einsum "bfhwpqrc->bcfphqwr" is a pure permutation; permute
        # exports as a clean ONNX Transpose (einsum creates a constant
        # initializer that breaks graphsurgeon external-data round-trip)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(
            x.shape[0],
            c,
            f * self.patch_size[0],
            h * self.patch_size[1],
            w * self.patch_size[2],
        )
        return x
