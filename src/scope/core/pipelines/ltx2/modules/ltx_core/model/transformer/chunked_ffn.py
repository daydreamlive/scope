"""Chunked FeedForward for memory-efficient inference.

This module provides a wrapper that processes FeedForward layers in chunks
along the sequence dimension, dramatically reducing peak memory usage.

The key insight is that FFN layers operate independently on each token position
(unlike attention which computes relationships across all positions). This means
FFN computation is embarrassingly parallel and can be chunked without any
mathematical difference in the output.

Memory Impact:
- Standard FFN: Creates intermediate tensor of shape (batch, seq_len, 4*dim)
  For seq_len=57000, dim=4096: ~3.7GB per layer
- Chunked FFN: Creates intermediate tensor of shape (batch, chunk_size, 4*dim)
  For chunk_size=4096, dim=4096: ~0.26GB per layer

With 96 FFN layers (48 video + 48 audio), this reduces peak activation memory
from ~50-60GB to ~5-6GB.

Reference:
- ComfyUI LTX-2 VRAM Memory Management:
  https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ltx_core.model.transformer.feed_forward import FeedForward

logger = logging.getLogger(__name__)


class ChunkedFeedForward(torch.nn.Module):
    """Wrapper that processes FeedForward in chunks to reduce memory.

    This wrapper intercepts the forward pass and processes the sequence
    in chunks, reducing peak memory from O(seq_len * 4*dim) to
    O(chunk_size * 4*dim).

    The result is mathematically identical to the original FFN - this is
    NOT an approximation. We're just trading memory for kernel launches.

    Args:
        original_ff: The original FeedForward module to wrap
        chunk_size: Number of tokens to process at once. Smaller = less memory
                   but more kernel launches. Default 4096 is a good balance.
    """

    def __init__(self, original_ff: FeedForward, chunk_size: int = 4096) -> None:
        super().__init__()
        self.ff = original_ff
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with chunked processing.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        seq_len = x.shape[1]

        # If sequence is small enough, just run normally
        if seq_len <= self.chunk_size:
            return self.ff(x)

        # Process in chunks along sequence dimension
        # Pre-allocate output tensor to avoid fragmentation
        output = torch.empty_like(x)

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            # Process chunk through original FFN
            output[:, start:end] = self.ff(x[:, start:end])

        return output

    def extra_repr(self) -> str:
        return f"chunk_size={self.chunk_size}"


def apply_chunked_ffn(
    model: torch.nn.Module,
    chunk_size: int = 4096,
    verbose: bool = True,
) -> int:
    """Apply chunked FFN to all FeedForward layers in a model.

    This function walks through the model and wraps all FeedForward layers
    with ChunkedFeedForward for memory-efficient inference.

    Args:
        model: The model to patch (typically LTXModel or its transformer)
        chunk_size: Chunk size for FFN processing
        verbose: Whether to log the number of layers patched

    Returns:
        Number of FeedForward layers that were wrapped
    """
    from ltx_core.model.transformer.feed_forward import FeedForward

    patched_count = 0

    # Find all FeedForward modules and their parent modules
    modules_to_patch: list[tuple[torch.nn.Module, str, FeedForward]] = []

    for name, module in model.named_modules():
        if isinstance(module, FeedForward) and not isinstance(
            module, ChunkedFeedForward
        ):
            # Find the parent module
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            modules_to_patch.append((parent, attr_name, module))

    # Apply patches
    for parent, attr_name, ff_module in modules_to_patch:
        wrapped = ChunkedFeedForward(ff_module, chunk_size=chunk_size)
        setattr(parent, attr_name, wrapped)
        patched_count += 1

    if verbose:
        logger.info(
            f"Applied chunked FFN to {patched_count} FeedForward layers "
            f"(chunk_size={chunk_size})"
        )

    return patched_count


def remove_chunked_ffn(model: torch.nn.Module, verbose: bool = True) -> int:
    """Remove chunked FFN wrappers, restoring original FeedForward layers.

    Args:
        model: The model to unpatch
        verbose: Whether to log the number of layers unpatched

    Returns:
        Number of ChunkedFeedForward wrappers that were removed
    """
    unpatched_count = 0

    # Find all ChunkedFeedForward modules
    modules_to_unpatch: list[tuple[torch.nn.Module, str, ChunkedFeedForward]] = []

    for name, module in model.named_modules():
        if isinstance(module, ChunkedFeedForward):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            modules_to_unpatch.append((parent, attr_name, module))

    # Remove wrappers
    for parent, attr_name, chunked_module in modules_to_unpatch:
        setattr(parent, attr_name, chunked_module.ff)
        unpatched_count += 1

    if verbose:
        logger.info(f"Removed chunked FFN from {unpatched_count} layers")

    return unpatched_count


def estimate_ffn_memory_savings(
    seq_len: int,
    dim: int = 4096,
    mult: int = 4,
    num_layers: int = 96,
    chunk_size: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Estimate memory savings from chunked FFN.

    Args:
        seq_len: Sequence length (number of tokens)
        dim: Hidden dimension
        mult: FFN expansion multiplier (typically 4)
        num_layers: Number of FFN layers (48 video + 48 audio = 96 for LTX-2)
        chunk_size: Chunk size for chunked processing
        dtype: Data type for tensors

    Returns:
        Dictionary with memory estimates in GB
    """
    bytes_per_element = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    inner_dim = dim * mult

    # Standard FFN: intermediate tensor is (batch=1, seq_len, inner_dim)
    standard_intermediate_bytes = seq_len * inner_dim * bytes_per_element
    standard_peak_gb = (standard_intermediate_bytes * num_layers) / (1024**3)

    # Chunked FFN: intermediate tensor is (batch=1, chunk_size, inner_dim)
    effective_chunk = min(chunk_size, seq_len)
    chunked_intermediate_bytes = effective_chunk * inner_dim * bytes_per_element
    # Only one chunk active at a time, but we have output tensor too
    chunked_peak_gb = (
        chunked_intermediate_bytes + seq_len * dim * bytes_per_element
    ) / (1024**3)

    return {
        "standard_peak_gb": standard_peak_gb,
        "chunked_peak_gb": chunked_peak_gb,
        "savings_gb": standard_peak_gb - chunked_peak_gb,
        "reduction_factor": standard_peak_gb / chunked_peak_gb
        if chunked_peak_gb > 0
        else float("inf"),
    }
