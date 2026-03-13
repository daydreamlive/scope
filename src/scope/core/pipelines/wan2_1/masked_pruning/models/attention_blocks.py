import math

import torch

from scope.core.pipelines.wan2_1.modules.attention import attention

from ..utils import M_DEGREE, causal_rope_apply_masked, select_m_degree_kv


def create_pruned_self_attn_class(base_self_attn_class):
    """Factory that creates a pruned self-attention class from any CausalWanSelfAttention.

    When prune_mask is None, delegates entirely to the parent's forward().
    When prune_mask is active, uses M-Degree Approximation attention:
      - causal_rope_apply_masked() on Q and K
      - select_m_degree_kv() for recent cache window
      - Attention: pruned Q x (cache window + pruned K/V)
      - Returns with {"action": "skip"} (no cache write)

    Args:
        base_self_attn_class: Any CausalWanSelfAttention class (not instance)

    Returns:
        A PrunedSelfAttention class that inherits from the given base
    """

    class PrunedSelfAttention(base_self_attn_class):
        """Self-attention with masked pruning support for M-Degree Approximation."""

        def forward(
            self,
            x,
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache=None,
            current_start=0,
            cache_start=None,
            sink_recache_after_switch=False,
            prune_mask=None,
        ):
            if prune_mask is None:
                # No pruning, delegate to parent
                return super().forward(
                    x,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    block_mask,
                    kv_cache,
                    current_start,
                    cache_start,
                    sink_recache_after_switch,
                )

            # M-Degree Approximation path: pruned attention
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
            if cache_start is None:
                cache_start = current_start

            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen

            num_frames = grid_sizes[0][0].item()
            full_mask = prune_mask.unsqueeze(0).expand(num_frames, -1).flatten()

            # Masked RoPE on both pruned Q and K
            roped_query = causal_rope_apply_masked(
                q,
                grid_sizes,
                freqs,
                start_frame=current_start_frame,
                spatial_mask=full_mask,
            ).type_as(v)
            roped_key = causal_rope_apply_masked(
                k,
                grid_sizes,
                freqs,
                start_frame=current_start_frame,
                spatial_mask=full_mask,
            ).type_as(v)

            # Select m recent frames from cache (RoPE'd K, raw V)
            sink_tokens = self.sink_size * frame_seqlen
            cache_k, cache_v = select_m_degree_kv(
                kv_cache, M_DEGREE, frame_seqlen, sink_tokens
            )

            # Attention: pruned Q x (cache window + pruned K/V)
            if cache_k.shape[1] > 0:
                attn_k = torch.cat([cache_k, roped_key], dim=1)
                attn_v = torch.cat([cache_v, v], dim=1)
            else:
                attn_k = roped_key
                attn_v = v

            x = attention(roped_query, attn_k, attn_v)

            # Output projection
            x = x.flatten(2)
            x = self.o(x)

            # Skip cache write (clean pass updates at full res)
            current_end = kv_cache["global_end_index"].item()
            local_end_index = kv_cache["local_end_index"].item()
            return x, (
                current_end,
                local_end_index,
                {"action": "skip", "is_recompute": True},
            )

    # Preserve original class name so LoRA target_modules discovery
    # (which matches on __class__.__name__) works transparently.
    PrunedSelfAttention.__name__ = base_self_attn_class.__name__
    PrunedSelfAttention.__qualname__ = base_self_attn_class.__qualname__
    return PrunedSelfAttention


def create_pruned_block_class(base_block_class):
    """Factory that creates a pruned attention block class from any CausalWanAttentionBlock.

    - Swaps self.self_attn with pruned version in __init__ (state_dict preserved)
    - Overrides forward() to compute frame_seqlen from prune_mask when active
    - Passes prune_mask to self_attn

    Args:
        base_block_class: Any CausalWanAttentionBlock class (not instance)

    Returns:
        A PrunedAttentionBlock class that inherits from the given base
    """

    # Get the self_attn class from the base block
    # We need to create the pruned self_attn class from the actual self_attn type
    # This is done at block init time since we need the instance's self_attn class

    class PrunedAttentionBlock(base_block_class):
        """Attention block with masked pruning support."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Create pruned self_attn class from the current self_attn's type
            PrunedSelfAttn = create_pruned_self_attn_class(type(self.self_attn))

            # Create new pruned self_attn with same config
            old_attn = self.self_attn
            new_attn = PrunedSelfAttn.__new__(PrunedSelfAttn)
            # Copy state from old to new
            new_attn.__dict__.update(old_attn.__dict__)
            new_attn.__class__ = PrunedSelfAttn
            self.self_attn = new_attn

        def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            cache_start=None,
            sink_recache_after_switch=False,
            prune_mask=None,
            **kwargs,
        ):
            num_frames = e.shape[1]
            # When pruning, frame_seqlen is the number of kept tokens per frame
            if prune_mask is not None:
                frame_seqlen = prune_mask.sum().item()
            else:
                frame_seqlen = x.shape[1] // num_frames

            e_mod = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

            # self-attention
            self_attn_result = self.self_attn(
                (
                    self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e_mod[1])
                    + e_mod[0]
                ).flatten(1, 2),
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                sink_recache_after_switch,
                prune_mask=prune_mask,
            )

            if kv_cache is not None:
                y, cache_update_info = self_attn_result
            else:
                y = self_attn_result
                cache_update_info = None

            x = x + (
                y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[2]
            ).flatten(1, 2)

            # cross-attention & ffn
            x = x + self.cross_attn(
                self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (
                    self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * (1 + e_mod[4])
                    + e_mod[3]
                ).flatten(1, 2)
            )
            x = x + (
                y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_mod[5]
            ).flatten(1, 2)

            if cache_update_info is not None:
                return x, cache_update_info
            else:
                return x

    # Preserve original class name so LoRA target_modules discovery
    # (which matches on __class__.__name__) works transparently.
    PrunedAttentionBlock.__name__ = base_block_class.__name__
    PrunedAttentionBlock.__qualname__ = base_block_class.__qualname__
    return PrunedAttentionBlock
