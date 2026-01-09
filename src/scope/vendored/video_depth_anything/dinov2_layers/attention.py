# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

import os
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


logger = logging.getLogger("dinov2")


try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("xFormers not available")
    XFORMERS_AVAILABLE = False


def _is_env_true(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or default).strip().lower() in ("1", "true", "yes", "on")


_USE_SDPA_FALLBACK = _is_env_true("SCOPE_VDA_SDPA_FALLBACK", default="1")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            assert attn_bias is None, "xFormers is required for nested tensors usage"
            # Fallback to PyTorch SDPA (Flash/MemEff/Math) when xFormers isn't available.
            # This is typically much faster than the naive q@k^T path on modern GPUs.
            if _USE_SDPA_FALLBACK and hasattr(F, "scaled_dot_product_attention"):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
                q, k, v = qkv.unbind(dim=2)  # (B, N, H, D)
                q = q.transpose(1, 2)  # (B, H, N, D)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                dropout_p = float(self.attn_drop.p) if self.training else 0.0
                try:
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=dropout_p,
                        is_causal=False,
                    )
                except Exception:
                    # If SDPA isn't available for this shape/dtype/device, fall back
                    # to the naive implementation to preserve correctness.
                    return super().forward(x)

                out = out.transpose(1, 2).reshape(B, N, C)
                out = self.proj(out)
                out = self.proj_drop(out)
                return out

            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

        
