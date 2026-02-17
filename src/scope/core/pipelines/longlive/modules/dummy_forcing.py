"""Dummy Forcing: three-group head classification for KV cache optimization.

Classifies attention heads into three groups based on their attention patterns:
  - Group A (first-frame heads): primarily attend to the first frame
  - Group B (local heads): primarily attend to recent/mid-range context
  - Group C (dummy heads): primarily attend to the current frame only

Each group gets the minimum context it needs, dramatically reducing total
attention computation while preserving quality.

Based on arXiv:2601.20499 (Guo et al., Microsoft Research / Tsinghua).

Performance: during classification, the cache's head dimension is reordered
so that groups occupy contiguous positions [C | A | B]. This allows all
subsequent per-group slices to be free contiguous views (no gather/copy).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from scope.core.pipelines.wan2_1.modules.attention import attention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Head classification
# ---------------------------------------------------------------------------


def compute_head_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    num_frames_per_block: int = 3,
) -> torch.Tensor:
    """Compute per-head attention distribution over first/mid/last regions.

    Mirrors the original ``online_head_classification`` from DummyForcing.

    Args:
        query: [B, L, num_heads, D] queries for the current block.
        key: [B, K, num_heads, D] all valid keys (cache + current).
        num_frames_per_block: frames per AR block (default 3 for LongLive).

    Returns:
        [3, B, num_heads] — attention fractions for (first, mid, last) regions.
    """
    B, L, H, D = query.shape
    HW = L // num_frames_per_block

    num_sampled = max(HW // 3, 1)
    sampled_idx = torch.randint(0, L, (num_sampled,), device=query.device)

    sampled_q = query[:, sampled_idx].transpose(1, 2).float()
    k_t = key.transpose(1, 2).float()

    scores = torch.matmul(sampled_q, k_t.transpose(-2, -1)) / (D**0.5)
    attn = F.softmax(scores, dim=-1)

    last_chunk = attn[:, :, :, -L:].sum(dim=-1).mean(dim=-1)
    mid_chunk = attn[:, :, :, HW:-L].sum(dim=-1).mean(dim=-1)
    first_chunk = attn[:, :, :, :HW].sum(dim=-1).mean(dim=-1)

    return torch.stack([first_chunk, mid_chunk, last_chunk])


def classify_heads(
    scores: torch.Tensor,
    num_dummy: int,
) -> list[dict[str, list[int]]]:
    """Classify heads into three groups (A, B, C) globally across all layers.

    Mirrors the original ``dynamic_head_programming`` from DummyForcing.

    - Group C (dummy): heads with lowest ``max(p_first, p_mid)`` — they don't
      need historical context.
    - Group A (first-frame): remaining heads where ``p_first >= p_mid``.
    - Group B (local/mid): remaining heads where ``p_mid > p_first``.

    Args:
        scores: [num_layers, num_heads, 3] attention scores (first, mid, last).
        num_dummy: total number of heads to classify as dummy (Group C).

    Returns:
        List of dicts (one per layer), each with keys ``group_a``, ``group_b``,
        ``group_c`` containing head index lists.
    """
    num_layers, num_heads, _ = scores.shape
    total_heads = num_layers * num_heads
    num_dummy = min(num_dummy, total_heads)

    p0_flat = scores[:, :, 0].reshape(-1)
    p1_flat = scores[:, :, 1].reshape(-1)
    p0_norm = p0_flat / (p0_flat.sum() + 1e-8)
    p1_norm = p1_flat / (p1_flat.sum() + 1e-8)

    cost = torch.maximum(p0_norm, p1_norm)
    sorted_indices = torch.argsort(cost)

    assignment = torch.zeros(total_heads, dtype=torch.long, device=scores.device)
    assignment[sorted_indices[:num_dummy]] = 2  # Group C

    remaining = (assignment != 2).nonzero(as_tuple=True)[0]
    for idx in remaining:
        assignment[idx] = 1 if p0_norm[idx] < p1_norm[idx] else 0

    assignment = assignment.reshape(num_layers, num_heads)

    head_groups = []
    n_a_total, n_b_total, n_c_total = 0, 0, 0
    for layer_idx in range(num_layers):
        ga = (assignment[layer_idx] == 0).nonzero(as_tuple=True)[0].tolist()
        gb = (assignment[layer_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        gc = (assignment[layer_idx] == 2).nonzero(as_tuple=True)[0].tolist()
        head_groups.append({"group_a": ga, "group_b": gb, "group_c": gc})
        n_a_total += len(ga)
        n_b_total += len(gb)
        n_c_total += len(gc)

    print(
        f"Dummy Forcing classification: A={n_a_total} B={n_b_total} C={n_c_total} "
        f"(total={total_heads})"
    )
    return head_groups


# ---------------------------------------------------------------------------
# Cache head reordering
# ---------------------------------------------------------------------------


def reorder_cache_heads(
    kv_cache: list[dict],
    head_groups: list[dict],
) -> None:
    """Reorder cache head dimension to [C | A | B] for contiguous slicing.

    After reordering, per-group dim-2 slices are free views (no gather/copy).
    Stores reorder indices, group sizes, and inverse reorder in each cache dict.
    """
    for layer_idx, cache in enumerate(kv_cache):
        groups = head_groups[layer_idx]
        gc = groups["group_c"]
        ga = groups["group_a"]
        gb = groups["group_b"]

        reorder = gc + ga + gb
        n_c = len(gc)
        n_a = len(ga)

        inv_reorder = [0] * len(reorder)
        for new_idx, old_idx in enumerate(reorder):
            inv_reorder[old_idx] = new_idx

        reorder_t = torch.tensor(reorder, device=cache["k"].device, dtype=torch.long)
        inv_reorder_t = torch.tensor(
            inv_reorder, device=cache["k"].device, dtype=torch.long
        )

        cache["k"] = cache["k"].index_select(2, reorder_t).contiguous()
        cache["v"] = cache["v"].index_select(2, reorder_t).contiguous()

        cache["df_head_groups"] = groups
        cache["df_n_c"] = n_c
        cache["df_n_a"] = n_a
        cache["df_reorder"] = reorder_t
        cache["df_inv_reorder"] = inv_reorder_t


def unreorder_cache_heads(kv_cache: list[dict]) -> None:
    """Restore original head ordering in the KV cache."""
    for cache in kv_cache:
        inv_reorder = cache.get("df_inv_reorder")
        if inv_reorder is not None:
            cache["k"] = cache["k"].index_select(2, inv_reorder).contiguous()
            cache["v"] = cache["v"].index_select(2, inv_reorder).contiguous()


# ---------------------------------------------------------------------------
# Three-path attention
# ---------------------------------------------------------------------------


def dummy_forcing_attention(
    roped_query: torch.Tensor,
    temp_k: torch.Tensor,
    temp_v: torch.Tensor,
    sink_tokens: int,
    frame_seqlen: int,
    num_new_tokens: int,
    local_end_index: int,
    local_context_length: int,
    n_c: int,
    n_a: int,
) -> torch.Tensor:
    """Three-path attention with contiguous head slices (cache heads pre-reordered).

    Expects heads in reordered layout: [0:n_c] = C, [n_c:n_c+n_a] = A, [n_c+n_a:] = B.
    All dim-2 slices are contiguous views.

    - Group C (dummy): most recent frame + current — minimal context.
    - Group A (first-frame): first frame from sink + current — anchoring context.
    - Group B (local): recent local window + current — motion context.

    Args:
        roped_query: [B, L, H, D] current queries (heads in reordered layout).
        temp_k: [B, cache_size, H, D] cache keys (heads reordered, current inserted).
        temp_v: [B, cache_size, H, D] cache values (heads reordered, current inserted).
        sink_tokens: number of persistent sink tokens at the front of cache.
        frame_seqlen: tokens per single latent frame (HW).
        num_new_tokens: number of current-frame tokens (already in temp_k).
        local_end_index: end index of valid data in temp_k/temp_v.
        local_context_length: number of AR blocks of recent context for Group B.
        n_c: number of Group C (dummy) heads.
        n_a: number of Group A (first-frame) heads.

    Returns:
        [B, L, H, D] attention output (heads in reordered layout).
    """
    n_total = roped_query.shape[2]
    n_b = n_total - n_c - n_a
    ca_boundary = n_c + n_a  # start of B heads

    results = []

    # ---- Group C (dummy): recent frame + current ----
    if n_c > 0:
        c_window = min(frame_seqlen + num_new_tokens, local_end_index)
        c_start = max(0, local_end_index - c_window)
        results.append(
            attention(
                roped_query[:, :, :n_c, :],
                temp_k[:, c_start:local_end_index, :n_c, :],
                temp_v[:, c_start:local_end_index, :n_c, :],
            )
        )

    # ---- Group A (first-frame): first frame + current ----
    if n_a > 0:
        a_q = roped_query[:, :, n_c:ca_boundary, :]
        # First frame from cache (beginning of sink region)
        a_k_first = temp_k[:, :frame_seqlen, n_c:ca_boundary, :]
        a_v_first = temp_v[:, :frame_seqlen, n_c:ca_boundary, :]
        # Current tokens (end of valid region)
        cur_start = max(0, local_end_index - num_new_tokens)
        a_k_cur = temp_k[:, cur_start:local_end_index, n_c:ca_boundary, :]
        a_v_cur = temp_v[:, cur_start:local_end_index, n_c:ca_boundary, :]
        results.append(
            attention(
                a_q,
                torch.cat([a_k_first, a_k_cur], dim=1),
                torch.cat([a_v_first, a_v_cur], dim=1),
            )
        )

    # ---- Group B (local): recent local window + current ----
    if n_b > 0:
        b_window = min(
            num_new_tokens + 3 * frame_seqlen * local_context_length,
            local_end_index,
        )
        b_start = max(0, local_end_index - b_window)
        results.append(
            attention(
                roped_query[:, :, ca_boundary:, :],
                temp_k[:, b_start:local_end_index, ca_boundary:, :],
                temp_v[:, b_start:local_end_index, ca_boundary:, :],
            )
        )

    return torch.cat(results, dim=2)
