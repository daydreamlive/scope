# Modified from https://github.com/NVlabs/LongLive
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from scope.core.pipelines.wan2_1.modules.attention import attention, compiled_attention
from .kv_cache_manager import apply_rope
from .model import (
    WAN_CROSSATTENTION_CLASSES,
    WanLayerNorm,
    WanRMSNorm,
    rope_params,
    sinusoidal_embedding_1d,
)


class CausalWanSelfAttention(nn.Module):
    """
    Self-attention with ring-buffer KV cache.

    This module contains ONLY tensor operations (no Python control flow,
    no .item() calls, no dict access). All cache management is handled
    externally by KVCacheManager.
    """

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6, **kwargs):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, cache_k, cache_v, write_indices, attn_mask, rope_freqs):
        """
        Pure tensor forward. No cache logic, no branching.

        Args:
            x: Input tensor [B, L, C].
            cache_k: This layer's key cache [B, cache_size, N, D].
            cache_v: This layer's value cache [B, cache_size, N, D].
            write_indices: Positions to write new K/V [L] (long tensor).
            attn_mask: Bool mask [1, 1, 1, cache_size] or None.
            rope_freqs: Precomputed RoPE freqs [L, 1, D//2] (complex).

        Returns:
            Output tensor [B, L, C].
        """
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim

        # QKV projection
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply precomputed RoPE
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Write new K/V into cache in-place (ring buffer position)
        cache_k[:, write_indices] = k
        cache_v[:, write_indices] = v

        # Attend query against full cache
        # Use fast SageAttention path when no mask needed (steady state),
        # fall back to SDPA when mask is required (filling phase only).
        if attn_mask is None:
            x = attention(q, cache_k, cache_v)
        else:
            x = compiled_attention(q, cache_k, cache_v, attn_mask=attn_mask)

        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    """
    Transformer block: self-attention + cross-attention + FFN with modulation.

    Pure tensor operations only. No cache logic, no conditional returns.
    """

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, qk_norm, eps)
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

    def forward(
        self,
        x,
        e,
        cache_k,
        cache_v,
        write_indices,
        attn_mask,
        rope_freqs,
        context,
        context_lens,
    ):
        """
        Args:
            x: [B, L, C] hidden states.
            e: [B, F, 6, C] time modulation embeddings.
            cache_k: [B, cache_size, N, D] key cache for this layer.
            cache_v: [B, cache_size, N, D] value cache for this layer.
            write_indices: [L] ring buffer write positions.
            attn_mask: [1, 1, 1, cache_size] or None.
            rope_freqs: [L, 1, D//2] precomputed RoPE (complex).
            context: [B, T, C] text embeddings.
            context_lens: [B] or None.
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # Self-attention with modulation
        y = self.self_attn(
            (
                self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                * (1 + e[1])
                + e[0]
            ).flatten(1, 2),
            cache_k,
            cache_v,
            write_indices,
            attn_mask,
            rope_freqs,
        )

        x = x + (
            y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]
        ).flatten(1, 2)

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # FFN with modulation
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

        return x


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
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            * (1 + e[1])
            + e[0]
        )
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.

    The forward pass is a pure tensor graph with no cache management logic.
    Cache is managed externally by KVCacheManager, which owns the cache_k/cache_v
    buffers and computes write_indices, attn_mask, rope_freqs before each call.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]
    _no_split_modules = ["CausalWanAttentionBlock"]
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

        # Persistent cache references (set by KVCacheManager)
        self.cache_k = None  # [num_layers, B, cache_size, N, D]
        self.cache_v = None  # [num_layers, B, cache_size, N, D]

    def set_cache(self, cache_k, cache_v):
        """Bind persistent cache buffers from KVCacheManager."""
        self.cache_k = cache_k
        self.cache_v = cache_v
        # Move freqs to the same device as the cache (once, not every forward)
        self.freqs = self.freqs.to(cache_k.device)

    def forward(self, x, t, context, write_indices, attn_mask, rope_freqs):
        """
        Clean forward pass. Pure tensor operations, suitable for torch.compile.

        All cache management (ring pointer, mask computation, RoPE precomputation)
        is handled externally by KVCacheManager before this call.

        Args:
            x: Input video tensor [B, C_in, F, H, W].
            t: Diffusion timesteps [B, F] or [B].
            context: Text embeddings [B, L_text, C_text].
            write_indices: Ring buffer write positions [num_new_tokens] (long).
            attn_mask: Bool attention mask [1, 1, 1, cache_size] or None.
            rope_freqs: Precomputed RoPE freqs [num_new_tokens, 1, D//2] (complex).

        Returns:
            Denoised video tensor [B, C_out, F, H, W].
        """
        # Patch embedding
        x = self.patch_embedding(x)
        grid_sizes = torch.tensor(x.shape[2:], dtype=torch.long)
        x = x.flatten(2).transpose(1, 2)

        # Time embedding
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )
        e0 = (
            self.time_projection(e)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=t.shape)
        )

        # Text embedding (pad to text_len if needed)
        context_lens = None
        if context.shape[1] < self.text_len:
            padded_context = torch.zeros(
                (context.shape[0], self.text_len, context.shape[2]),
                dtype=context.dtype,
                device=context.device,
            )
            padded_context[:, : context.shape[1], :] = context
            context = padded_context
        context = self.text_embedding(context)

        # Transformer blocks - pure tensor loop, no cache logic
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                e0,
                self.cache_k[i],
                self.cache_v[i],
                write_indices,
                attn_mask,
                rope_freqs,
                context,
                context_lens,
            )

        # Head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Batch of patchified features, shape [B, L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensor with shape [B, C_out, F, H / 8, W / 8]
        """
        c = self.out_dim
        v = grid_sizes.tolist()
        x = x.view(x.shape[0], *v, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(x.shape[0], c, *[i * j for i, j in zip(v, self.patch_size)])
        return x
