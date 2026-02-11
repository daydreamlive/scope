"""
Ring buffer KV cache manager for CausalWanModel.

Manages KV cache as a ring buffer with fixed sink tokens and a circular
overwrite region. All cache management logic lives here; the compiled
model forward pass is pure tensor operations.
"""

import torch


def precompute_rope_freqs(freqs, f, h, w, start_frame=0):
    """
    Precompute RoPE frequency tensor for a given grid and starting frame.

    This is called OUTSIDE the compiled graph by the cache manager.

    Args:
        freqs: Full RoPE frequency table, shape [max_seq_len, head_dim // 2] (complex).
        f: Number of frames (temporal grid size).
        h: Height grid size (after patching).
        w: Width grid size (after patching).
        start_frame: Starting frame index for temporal frequencies.

    Returns:
        freqs_i: Precomputed frequency tensor, shape [seq_len, 1, head_dim // 2] (complex).
    """
    c = freqs.shape[1]
    freq_parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    seq_len = f * h * w
    freqs_i = torch.cat(
        [
            freq_parts[0][start_frame : start_frame + f]
            .view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            freq_parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freq_parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_i


def apply_rope(x, rope_freqs):
    """
    Apply precomputed RoPE frequencies to Q or K tensor.

    This is called INSIDE the compiled graph; pure tensor ops only.

    Args:
        x: Tensor of shape [B, seq_len, num_heads, head_dim].
        rope_freqs: Precomputed freqs of shape [seq_len, 1, head_dim // 2] (complex).

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    seq_len = rope_freqs.shape[0]
    n = x.size(2)
    c = x.size(3) // 2

    x_complex = torch.view_as_complex(
        x[:, :seq_len].to(torch.float64).reshape(-1, seq_len, n, c, 2)
    )
    x_rotated = torch.view_as_real(x_complex * rope_freqs).flatten(3)

    # If x has tokens beyond seq_len, preserve them
    if x.shape[1] > seq_len:
        x_rotated = torch.cat([x_rotated, x[:, seq_len:]], dim=1)

    return x_rotated.type_as(x)


class KVCacheManager:
    """
    Ring buffer KV cache manager.

    Cache layout: [SINK (fixed) | RING (circular overwrite)]

    The sink region (positions 0 to sink_tokens) is written once during the
    first step and never overwritten. The ring region (positions sink_tokens
    to cache_size) is a circular buffer that overwrites the oldest entries.

    This class owns the cache tensors and computes per-step metadata
    (write_indices, attn_mask, rope_freqs) that are passed to the compiled
    model forward.
    """

    def __init__(
        self,
        num_layers,
        batch_size,
        cache_size,
        num_heads,
        head_dim,
        sink_tokens,
        frame_seqlen,
        device,
        dtype,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sink_tokens = sink_tokens
        self.frame_seqlen = frame_seqlen
        self.device = device
        self.dtype = dtype

        self.ring_capacity = cache_size - sink_tokens

        # Stacked cache tensors: [num_layers, B, cache_size, num_heads, head_dim]
        self.cache_k = torch.zeros(
            num_layers, batch_size, cache_size, num_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.cache_v = torch.zeros(
            num_layers, batch_size, cache_size, num_heads, head_dim,
            dtype=dtype, device=device,
        )

        # Ring buffer state
        self.ring_ptr = 0
        self.ring_filled = False
        self.num_valid_tokens = 0

    def prepare_step(self, num_new_tokens, start_frame, freqs, f, h, w):
        """
        Compute write_indices, attn_mask, and rope_freqs for the current step.

        Called BEFORE the compiled model forward.

        Args:
            num_new_tokens: Number of new tokens being processed this step.
            start_frame: Starting frame index (for RoPE computation).
            freqs: Full RoPE frequency table (complex tensor).
            f: Number of frames in this step.
            h: Height grid size.
            w: Width grid size.

        Returns:
            write_indices: Tensor [num_new_tokens] with positions to write in cache.
            attn_mask: Bool tensor [1, 1, 1, cache_size] or None if all valid.
            rope_freqs: Complex tensor [seq_len, 1, head_dim//2] for RoPE.
        """
        if self.ring_ptr < self.sink_tokens:
            # Sink fill phase: write linearly from position 0
            write_indices = torch.arange(
                self.ring_ptr, self.ring_ptr + num_new_tokens,
                device=self.device, dtype=torch.long,
            )
        else:
            # Ring phase: wrap indices within [sink_tokens, cache_size)
            offsets = torch.arange(num_new_tokens, device=self.device, dtype=torch.long)
            write_indices = self.sink_tokens + (
                (self.ring_ptr - self.sink_tokens + offsets) % self.ring_capacity
            )

        # Determine if all cache positions will be valid after write
        fills_ring = num_new_tokens >= self.ring_capacity
        if self.ring_filled or fills_ring:
            attn_mask = None
        elif self.ring_ptr < self.sink_tokens:
            # Sink fill: positions [0, ring_ptr + num_new_tokens) are valid
            valid_end = self.ring_ptr + num_new_tokens
            attn_mask = torch.zeros(
                1, 1, 1, self.cache_size,
                dtype=torch.bool, device=self.device,
            )
            attn_mask[..., :valid_end] = True
        else:
            # Ring filling (not yet wrapped): [0, ring_ptr + num_new_tokens) valid
            valid_end = self.ring_ptr + num_new_tokens
            if valid_end >= self.cache_size:
                attn_mask = None
            else:
                attn_mask = torch.zeros(
                    1, 1, 1, self.cache_size,
                    dtype=torch.bool, device=self.device,
                )
                attn_mask[..., :valid_end] = True

        rope_freqs = precompute_rope_freqs(freqs, f, h, w, start_frame)

        return write_indices, attn_mask, rope_freqs

    def advance(self, num_new_tokens):
        """
        Advance the ring pointer after a forward pass completes.

        Args:
            num_new_tokens: Number of tokens that were written this step.
        """
        if self.ring_ptr < self.sink_tokens:
            # Was in sink fill phase
            self.ring_ptr += num_new_tokens
            self.num_valid_tokens = self.ring_ptr
        else:
            # Ring phase: advance with wrapping
            self.ring_ptr = self.sink_tokens + (
                (self.ring_ptr - self.sink_tokens + num_new_tokens) % self.ring_capacity
            )
            self.num_valid_tokens = min(
                self.num_valid_tokens + num_new_tokens, self.cache_size
            )
            if num_new_tokens >= self.ring_capacity:
                self.ring_filled = True
            elif self.num_valid_tokens >= self.cache_size:
                self.ring_filled = True

    def reset_for_recache(self, keep_sink=True):
        """
        Reset the cache for recaching.

        Args:
            keep_sink: If True (global_sink mode), preserve sink tokens and
                       only zero the ring portion. If False, zero everything.
        """
        if keep_sink and self.sink_tokens > 0:
            self.cache_k[:, :, self.sink_tokens:].zero_()
            self.cache_v[:, :, self.sink_tokens:].zero_()
            self.ring_ptr = self.sink_tokens
            self.num_valid_tokens = self.sink_tokens
        else:
            self.cache_k.zero_()
            self.cache_v.zero_()
            self.ring_ptr = 0
            self.num_valid_tokens = 0
        self.ring_filled = False

    def full_reset(self):
        """Reset the entire cache and all state (e.g. on hard cut)."""
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.ring_ptr = 0
        self.ring_filled = False
        self.num_valid_tokens = 0
