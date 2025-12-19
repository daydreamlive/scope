# Modified from notes/VACE/vace/models/wan/modules/model.py
# Adapted for causal/autoregressive generation
# Pipeline-agnostic using duck typing with factory pattern

import torch
import torch.nn as nn


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
            reference_prefix_length=0,
            **kwargs,
        ):
            """
            Forward pass with optional VACE hint injection.

            Args:
                hints: List of VACE hints, one per injection layer
                context_scale: Scaling factor for hint injection
                reference_prefix_length: Number of tokens in reference "frame" (for hint alignment)
                **kwargs: Pipeline-specific parameters (kv_cache, crossattn_cache, etc.)
            """
            # Standard forward pass
            # Reference tokens are now a proper "frame" so they go through normally
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
            # Hints apply to video tokens only (skip reference prefix)
            if hints is not None and self.block_id is not None:
                hint = hints[self.block_id]

                # Extract video portion (skip reference prefix)
                if reference_prefix_length > 0:
                    video_tokens = x[:, reference_prefix_length:, :]
                else:
                    video_tokens = x

                # Slice hint to match video token count (x is unpadded, hint may be padded)
                if hint.shape[1] > video_tokens.shape[1]:
                    hint = hint[:, : video_tokens.shape[1], :]

                # Apply hint injection to video tokens
                video_tokens = video_tokens + hint * context_scale

                # Reconstruct sequence
                if reference_prefix_length > 0:
                    x = torch.cat(
                        [x[:, :reference_prefix_length, :], video_tokens], dim=1
                    )
                else:
                    x = video_tokens

            # Return with cache info if applicable
            if cache_update_info is not None:
                return x, cache_update_info
            else:
                return x

    return BaseWanAttentionBlock
