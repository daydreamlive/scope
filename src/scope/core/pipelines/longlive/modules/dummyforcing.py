import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DummyForcingConfig:
    num_dummy: int
    ar_start: int
    local_context_length: int
    last_timestep: int = 250


def online_head_classification(
    query: torch.Tensor, key: torch.Tensor, ar_step: int
) -> torch.Tensor:
    B, L, head, C = query.shape
    HW = L // 3 if ar_step == 1 else L
    num_sampled_rows = HW // 3
    sampled_rows = torch.randint(low=0, high=L, size=(num_sampled_rows,))
    sampled_q = query[:, sampled_rows].transpose(1, 2)
    key_t = key.transpose(1, 2)
    sampled_qk_scores = torch.matmul(sampled_q, key_t.transpose(-2, -1)) / (C**0.5)
    sampled_attn_weights = F.softmax(sampled_qk_scores, dim=-1)
    last_chunk_agg = sampled_attn_weights[:, :, :, -L:].sum(dim=-1).mean(dim=-1)
    mid_chunk_agg = sampled_attn_weights[:, :, :, HW:-L].sum(dim=-1).mean(dim=-1)
    first_chunk_agg = sampled_attn_weights[:, :, :, :HW].sum(dim=-1).mean(dim=-1)
    return torch.stack([first_chunk_agg, mid_chunk_agg, last_chunk_agg])


def dynamic_head_programming(
    probs: torch.Tensor, num_dummy: int = 180
) -> tuple[dict, dict, dict]:
    num_layer, num_head, _ = probs.shape
    p0_flat = probs[:, :, 0].reshape(-1)
    p1_flat = probs[:, :, 1].reshape(-1)
    p0_norm = p0_flat / p0_flat.sum()
    p1_norm = p1_flat / p1_flat.sum()
    cost = torch.maximum(p0_norm, p1_norm)
    sorted_indices = torch.argsort(cost)
    c_indices_flat = sorted_indices[:num_dummy]
    assignment = torch.zeros(num_layer * num_head, dtype=torch.long)
    assignment[c_indices_flat] = 2

    remaining_mask = assignment != 2
    remaining_indices = torch.nonzero(remaining_mask, as_tuple=True)[0]
    for idx in remaining_indices:
        if p0_norm[idx] < p1_norm[idx]:
            assignment[idx] = 1
        else:
            assignment[idx] = 0

    assignment = assignment.reshape(num_layer, num_head)
    group_a: dict[int, list[int]] = {}
    group_b: dict[int, list[int]] = {}
    group_c: dict[int, list[int]] = {}
    for layer_idx in range(num_layer):
        group_a[layer_idx] = (
            (assignment[layer_idx] == 0).nonzero(as_tuple=True)[0].tolist()
        )
        group_b[layer_idx] = (
            (assignment[layer_idx] == 1).nonzero(as_tuple=True)[0].tolist()
        )
        group_c[layer_idx] = (
            (assignment[layer_idx] == 2).nonzero(as_tuple=True)[0].tolist()
        )
    return group_a, group_b, group_c


def heterogeneous_memory_allocation(
    kv_cache: list[dict],
    num_dummy: int,
    frame_seqlen: int,
    local_context_length: int,
):
    global_frame_attn_score = torch.stack(
        [layer_info["frame_attn_score"][:, 0] for layer_info in kv_cache]
    ).transpose(1, 2)
    global_group_first, global_group_mid, global_group_last = dynamic_head_programming(
        global_frame_attn_score, num_dummy
    )

    HW = frame_seqlen
    for layer_idx in range(len(kv_cache)):
        group_first = global_group_first[layer_idx]
        group_mid = global_group_mid[layer_idx]
        group_last = global_group_last[layer_idx]
        cur_cache = kv_cache[layer_idx]

        unified_k = cur_cache["k"]
        unified_v = cur_cache["v"]
        local_end = cur_cache["local_end_index"].item()

        sink_region_k = unified_k[:, :HW]
        sink_region_v = unified_v[:, :HW]
        local_region_k = unified_k[:, HW:local_end]
        local_region_v = unified_v[:, HW:local_end]

        cur_cache["sink_k"] = (
            torch.cat(
                [
                    sink_region_k[:, :, group_first],
                    local_region_k[:, -HW:, group_last],
                ],
                dim=2,
            )
            .contiguous()
            .clone()
        )
        cur_cache["sink_v"] = (
            torch.cat(
                [
                    sink_region_v[:, :, group_first],
                    local_region_v[:, -HW:, group_last],
                ],
                dim=2,
            )
            .contiguous()
            .clone()
        )

        max_local_tokens = 3 * HW * local_context_length
        cur_cache["local_k"] = (
            local_region_k[:, -max_local_tokens:, group_mid].contiguous().clone()
        )
        cur_cache["local_v"] = (
            local_region_v[:, -max_local_tokens:, group_mid].contiguous().clone()
        )

        cur_cache["headgroup_first"] = group_first
        cur_cache["headgroup_mid"] = group_mid
        cur_cache["headgroup_last"] = group_last
        cur_cache["dummy_forcing_active"] = True

    num_layers = len(kv_cache)
    total_heads = num_layers * len(global_frame_attn_score[0])
    n_a = sum(len(global_group_first[i]) for i in range(num_layers))
    n_b = sum(len(global_group_mid[i]) for i in range(num_layers))
    n_c = sum(len(global_group_last[i]) for i in range(num_layers))
    logger.info(
        "[DummyForcing] Activated: sink=%d neighbor=%d dummy=%d (of %d total heads)",
        n_a,
        n_b,
        n_c,
        total_heads,
    )
