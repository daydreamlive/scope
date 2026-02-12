# Modified from https://github.com/ali-vilab/VACE/blob/48eb44f1c4be87cc65a98bff985a26976841e9f3/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation
# Pipeline-agnostic using duck typing with factory pattern

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VaceSelfAttention(nn.Module):
    """Drop-in replacement for CausalWanSelfAttention optimized for VACE blocks.

    VACE blocks don't use KV cache, causal masking, or block_mask features.
    This replaces flex_attention + padding-to-128 with a simple SDPA call,
    eliminating:
    - Padding to multiple of 128 (3 tensor allocations + 3 cat ops per call)
    - flex_attention compiled graph overhead
    - block_mask processing
    - .item() GPU-CPU sync for is_tf check

    Uses the same Q/K/V/O linear layers and RoPE as the original.
    """

    def __init__(self, original_self_attn, rope_apply_fn, flash_attention_fn):
        super().__init__()
        self.num_heads = original_self_attn.num_heads
        self.head_dim = original_self_attn.head_dim
        # Share the same linear layers (no extra parameters)
        self.q = original_self_attn.q
        self.k = original_self_attn.k
        self.v = original_self_attn.v
        self.o = original_self_attn.o
        self.norm_q = original_self_attn.norm_q
        self.norm_k = original_self_attn.norm_k
        self._rope_apply = rope_apply_fn
        self._flash_attention = flash_attention_fn

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask=None,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply RoPE
        q = self._rope_apply(q, grid_sizes, freqs).type_as(v)
        k = self._rope_apply(k, grid_sizes, freqs).type_as(v)

        # flash_attention expects (B, S, num_heads, head_dim) — already in that layout
        # No causal mask needed for VACE — bidirectional attention
        x = self._flash_attention(q, k, v)

        # Output projection
        x = self.o(x.flatten(2))

        return x


def swap_vace_self_attention(vace_blocks, rope_apply_fn=None, flash_attention_fn=None):
    """Replace CausalWanSelfAttention with VaceSelfAttention on VACE blocks.

    Args:
        vace_blocks: nn.ModuleList of VACE attention blocks
        rope_apply_fn: The rope_apply function from the pipeline's model module.
            If None, auto-discovers from the self_attn module's globals.
        flash_attention_fn: The flash_attention function. If None, auto-discovers
            from the module that imports it (typically the cross-attention module).
    """
    if rope_apply_fn is None or flash_attention_fn is None:
        # Auto-discover from the module where self_attn's parent block class is defined
        first_block = vace_blocks[0]
        if hasattr(first_block, "self_attn"):
            import sys

            # rope_apply lives in the same module as the block's base class
            # Walk MRO to find a module with rope_apply
            for cls in type(first_block).__mro__:
                mod = sys.modules.get(cls.__module__)
                if mod is None:
                    continue
                if rope_apply_fn is None and hasattr(mod, "rope_apply"):
                    rope_apply_fn = mod.rope_apply
                if flash_attention_fn is None and hasattr(mod, "flash_attention"):
                    flash_attention_fn = mod.flash_attention

    if rope_apply_fn is None:
        logger.warning(
            "swap_vace_self_attention: Could not auto-discover rope_apply, skipping swap"
        )
        return

    if flash_attention_fn is None:
        # Fallback: import from the shared attention module
        try:
            from scope.core.pipelines.wan2_1.modules.attention import flash_attention

            flash_attention_fn = flash_attention
        except ImportError:
            logger.warning(
                "swap_vace_self_attention: Could not find flash_attention, skipping swap"
            )
            return

    for block in vace_blocks:
        if hasattr(block, "self_attn"):
            block.self_attn = VaceSelfAttention(
                block.self_attn, rope_apply_fn, flash_attention_fn
            )
    logger.info(
        f"Swapped self-attention on {len(vace_blocks)} VACE blocks "
        f"to VaceSelfAttention (flash_attn)"
    )


def create_vace_attention_block_class(base_attention_block_class):
    """
    Factory that creates a VaceWanAttentionBlock class inheriting from any CausalWanAttentionBlock.

    Uses duck typing - assumes base_attention_block_class has:
    - self.dim attribute
    - forward() method with standard transformer block signature

    Args:
        base_attention_block_class: Any CausalWanAttentionBlock class (not instance)

    Returns:
        A VaceWanAttentionBlock class that inherits from the given base
    """

    class VaceWanAttentionBlock(base_attention_block_class):
        """VACE attention block with zero-initialized projection layers for hint injection."""

        def __init__(self, *args, block_id=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.block_id = block_id

            # Initialize projection layers for hint accumulation
            # Duck typing: assume self.dim exists from base class
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
            # Pass block_mask to ensure the non-caching attention path is used
            # crossattn_cache=None because VACE processes reference images without caching
            c = super().forward(
                c,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                block_mask,  # Must pass block_mask to trigger non-caching path in KREA
                kv_cache=None,
                crossattn_cache=None,  # VACE doesn't use cross-attention caching
                current_start=0,
            )

            # Handle case where block returns tuple (shouldn't happen with kv_cache=None)
            if isinstance(c, tuple):
                c = c[0]

            # Generate hint for injection
            c_skip = self.after_proj(c)

            all_c += [c_skip, c]

            # Stack and return
            return torch.stack(all_c)

    return VaceWanAttentionBlock


def create_base_attention_block_class(base_attention_block_class):
    """
    Factory that creates a BaseWanAttentionBlock class with hint injection support.

    Uses duck typing - assumes base_attention_block_class has:
    - forward() method with standard transformer block signature

    Args:
        base_attention_block_class: Any CausalWanAttentionBlock class (not instance)

    Returns:
        A BaseWanAttentionBlock class that inherits from the given base
    """

    class BaseWanAttentionBlock(base_attention_block_class):
        """Base attention block with VACE hint injection support."""

        def __init__(self, *args, block_id=None, **kwargs):
            super().__init__(*args, **kwargs)
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
            **kwargs,
        ):
            """
            Forward pass with optional VACE hint injection.

            Args:
                hints: List of VACE hints, one per injection layer
                context_scale: Scaling factor for hint injection
                **kwargs: Pipeline-specific parameters (kv_cache, crossattn_cache, etc.)
            """
            # Standard forward pass
            result = super().forward(
                x,
                e,
                seq_lens,
                grid_sizes,
                freqs,
                context,
                context_lens,
                block_mask,
                **kwargs,
            )

            # Handle cache updates if present
            if isinstance(result, tuple):
                x, cache_update_info = result
            else:
                x = result
                cache_update_info = None

            # Inject VACE hint if this block has one
            if hints is not None and self.block_id is not None:
                hint = hints[self.block_id]
                # Slice hint to match x's sequence length (x is unpadded, hint may be padded)
                if hint.shape[1] > x.shape[1]:
                    hint = hint[:, : x.shape[1], :]

                # Apply context scale (default to 1.0 if None for safety)
                scale = context_scale if context_scale is not None else 1.0
                x = x + hint * scale

            # Return with cache info if applicable
            if cache_update_info is not None:
                return x, cache_update_info
            else:
                return x

    return BaseWanAttentionBlock
