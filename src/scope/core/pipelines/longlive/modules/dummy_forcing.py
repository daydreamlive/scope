"""Dummy Forcing: binary head classification for KV cache optimization.

Identifies 'dummy' attention heads that primarily attend to the current frame
and routes them through a shorter attention path (sink + current only),
reducing attention computation while maintaining quality.

Based on arXiv:2601.20499 (Guo et al., Microsoft Research / Tsinghua).

Simplified to binary classification (dummy vs. normal) for integration
with the LongLive pipeline's existing cache management. Cache writes are
uniform for all heads; only the read path differs per group.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from scope.core.pipelines.wan2_1.modules.attention import attention

logger = logging.getLogger(__name__)


def compute_df_score(
    query: torch.Tensor,
    key: torch.Tensor,
    num_current_tokens: int,
) -> torch.Tensor:
    """Compute per-head current-frame attention fraction for dummy head classification.

    Measures how much each attention head attends to the current frame tokens
    vs. historical cache tokens. Heads with high scores are candidates for
    dummy classification (they mostly attend to the current frame and don't
    benefit from a long KV cache).

    Args:
        query: [B, L, num_heads, D] queries for the current block.
        key: [B, K, num_heads, D] all valid keys (cache + current).
        num_current_tokens: number of tokens at the end of key that belong
            to the current frame (= query length L).

    Returns:
        Tensor of shape [num_heads] with current-frame attention fraction per head.
    """
    B, L, H, D = query.shape

    num_sampled = max(L // 3, 1)
    sampled_idx = torch.randint(0, L, (num_sampled,), device=query.device)

    sampled_q = query[:, sampled_idx].transpose(1, 2).float()
    k_t = key.transpose(1, 2).float()

    scores = torch.matmul(sampled_q, k_t.transpose(-2, -1)) / (D**0.5)
    attn_weights = F.softmax(scores, dim=-1)

    current_attn = attn_weights[:, :, :, -num_current_tokens:].sum(dim=-1).mean(dim=-1)
    return current_attn[0]


def select_dummy_heads(
    scores: torch.Tensor,
    num_dummy: int,
) -> list[dict[str, list[int]]]:
    """Select dummy heads globally across all layers.

    Heads with the highest current-frame attention fraction are classified
    as dummy. The classification is global (across all layers) so that the
    total number of dummy heads matches ``num_dummy``.

    Args:
        scores: [num_layers, num_heads] current-frame attention scores.
        num_dummy: total number of heads to classify as dummy.

    Returns:
        List of dicts (one per layer), each with keys:
            ``dummy_heads``: list of head indices classified as dummy.
            ``normal_heads``: list of head indices classified as normal.
    """
    num_layers, num_heads = scores.shape
    total_heads = num_layers * num_heads
    num_dummy = min(num_dummy, total_heads)

    flat_scores = scores.reshape(-1)
    _, sorted_indices = torch.sort(flat_scores, descending=True)

    is_dummy = torch.zeros(total_heads, dtype=torch.bool, device=scores.device)
    is_dummy[sorted_indices[:num_dummy]] = True
    is_dummy = is_dummy.reshape(num_layers, num_heads)

    head_groups = []
    for layer_idx in range(num_layers):
        dummy = is_dummy[layer_idx].nonzero(as_tuple=True)[0].tolist()
        normal = (~is_dummy[layer_idx]).nonzero(as_tuple=True)[0].tolist()
        head_groups.append({"dummy_heads": dummy, "normal_heads": normal})

    logger.info(
        "Dummy Forcing classification complete: %d/%d heads classified as dummy",
        num_dummy,
        total_heads,
    )

    return head_groups


def dummy_forcing_attention(
    roped_query: torch.Tensor,
    roped_key: torch.Tensor,
    v: torch.Tensor,
    temp_k: torch.Tensor,
    temp_v: torch.Tensor,
    sink_tokens: int,
    local_end_index: int,
    max_attention_size: int,
    head_groups: dict[str, list[int]],
) -> torch.Tensor:
    """Two-path attention: normal heads use full window, dummy heads use sink only.

    Normal heads run the standard longlive attention (sink + local window).
    Dummy heads only attend to sink tokens + current input tokens, skipping
    the full rolling window. Cache writes are handled identically for all
    heads by the caller; only the read path differs here.

    Args:
        roped_query: [B, L, H, D] current queries with RoPE applied.
        roped_key: [B, L, H, D] current keys with RoPE applied.
        v: [B, L, H, D] current values (no RoPE).
        temp_k: [B, cache_size, H, D] cache keys with current tokens inserted.
        temp_v: [B, cache_size, H, D] cache values with current tokens inserted.
        sink_tokens: number of persistent sink tokens at the front of cache.
        local_end_index: end index of valid data in temp_k/temp_v.
        max_attention_size: maximum attention window size for normal heads.
        head_groups: dict with ``dummy_heads`` and ``normal_heads`` index lists.

    Returns:
        [B, L, H, D] attention output for all heads.
    """
    normal_heads = head_groups["normal_heads"]
    dummy_heads = head_groups["dummy_heads"]

    out = torch.empty_like(roped_query)

    # Path 1: Normal heads — full local window (standard longlive attention)
    if len(normal_heads) > 0:
        normal_q = roped_query[:, :, normal_heads, :]

        if sink_tokens > 0:
            local_budget = max_attention_size - sink_tokens
            k_sink = temp_k[:, :sink_tokens, normal_heads, :]
            v_sink = temp_v[:, :sink_tokens, normal_heads, :]

            if local_budget > 0:
                local_start = max(sink_tokens, local_end_index - local_budget)
                k_local = temp_k[:, local_start:local_end_index, normal_heads, :]
                v_local = temp_v[:, local_start:local_end_index, normal_heads, :]
                k_cat = torch.cat([k_sink, k_local], dim=1)
                v_cat = torch.cat([v_sink, v_local], dim=1)
            else:
                k_cat = k_sink
                v_cat = v_sink

            out[:, :, normal_heads, :] = attention(normal_q, k_cat, v_cat)
        else:
            window_start = max(0, local_end_index - max_attention_size)
            out[:, :, normal_heads, :] = attention(
                normal_q,
                temp_k[:, window_start:local_end_index, normal_heads, :],
                temp_v[:, window_start:local_end_index, normal_heads, :],
            )

    # Path 2: Dummy heads — sink + current only (skip the rolling window)
    if len(dummy_heads) > 0:
        dummy_q = roped_query[:, :, dummy_heads, :]
        dummy_k = torch.cat(
            [
                temp_k[:, :sink_tokens, dummy_heads, :],
                roped_key[:, :, dummy_heads, :],
            ],
            dim=1,
        )
        dummy_v = torch.cat(
            [
                temp_v[:, :sink_tokens, dummy_heads, :],
                v[:, :, dummy_heads, :],
            ],
            dim=1,
        )
        out[:, :, dummy_heads, :] = attention(dummy_q, dummy_k, dummy_v)

    return out
