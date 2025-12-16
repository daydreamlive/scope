# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation with Longlive

import torch
import torch.nn as nn

from ....longlive.modules.causal_model import CausalWanAttentionBlock


class VaceWanAttentionBlock(CausalWanAttentionBlock):
    """VACE attention block with zero-initialized projection layers for hint injection."""

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
        block_id=0,
    ):
        super().__init__(
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
        self.block_id = block_id

        # Initialize projection layers for hint accumulation
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)

        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward_vace(
        self,
        c,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        crossattn_cache=None,
    ):
        """
        Forward pass for VACE blocks.

        Args:
            c: Accumulated VACE context from previous blocks (stacked hints + current)
            x: Input latent features
            Other args: Standard transformer block arguments

        Returns:
            Updated VACE context stack with new hint appended
        """
        # Unpack accumulated hints
        if self.block_id == 0:
            # c is padded to seq_len, but x may be shorter (unpadded for causal KV cache)
            # Slice c to match x's size for residual addition
            c_sliced = c[:, : x.size(1), :]

            before_proj_out = self.before_proj(c_sliced)
            c = before_proj_out + x

            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        # Run standard transformer block on current context
        # VACE blocks don't use caching since they process reference images once
        c = super().forward(
            c,
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
        )

        # Generate hint for injection
        c_skip = self.after_proj(c)

        all_c += [c_skip, c]

        # Stack and return
        return torch.stack(all_c)


class BaseWanAttentionBlock(CausalWanAttentionBlock):
    """Base attention block with VACE hint injection support."""

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
        block_id=None,
    ):
        super().__init__(
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
        self.block_id = block_id

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
        hints=None,
        context_scale=1.0,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
    ):
        """
        Forward pass with optional VACE hint injection.

        Args:
            hints: List of VACE hints, one per injection layer
            context_scale: Scaling factor for hint injection
        """
        result = super().forward(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            kv_cache,
            crossattn_cache,
            current_start,
            cache_start,
        )

        # Handle cache updates if present
        if kv_cache is not None and isinstance(result, tuple):
            x, cache_update_info = result
        else:
            x = result
            cache_update_info = None

        # Inject VACE hint if this block has one
        if hints is not None and self.block_id is not None:
            hint = hints[self.block_id]
            # Slice hint to match x's sequence length (x is unpadded, hint may be padded to seq_len)
            if hint.shape[1] > x.shape[1]:
                hint = hint[:, : x.shape[1], :]
            x = x + hint * context_scale

        # Return with cache info if applicable
        if cache_update_info is not None:
            return x, cache_update_info
        else:
            return x
