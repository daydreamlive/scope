"""Weight streaming utilities for memory-efficient inference.

This module provides block-level weight streaming that enables running large
transformer models on GPUs with limited VRAM by:

1. Keeping only a subset of transformer blocks on GPU at any time
2. Streaming blocks from CPU to GPU during the forward pass
3. Using CUDA streams for async prefetching to hide transfer latency
4. Using pinned memory for faster CPU<->GPU transfers

Based on techniques from ComfyUI and Kijai's implementations.

Usage:
    from scope.core.pipelines.weight_streaming import (
        BlockStreamingConfig,
        setup_block_streaming,
        cleanup_block_streaming,
    )

    # Configure streaming
    config = BlockStreamingConfig(
        blocks_to_stream=24,  # Keep 24 blocks on CPU, stream during forward
        prefetch_blocks=1,    # Prefetch 1 block ahead
        use_pinned_memory=True,
    )

    # Apply to model
    setup_block_streaming(model.transformer_blocks, config)

    # Run inference normally - streaming happens automatically
    output = model(input)

    # Cleanup when done
    cleanup_block_streaming(model.transformer_blocks)
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class BlockStreamingConfig:
    """Configuration for block-level weight streaming.

    Attributes:
        blocks_to_stream: Number of transformer blocks to keep on CPU and stream
            during forward pass. Set to 0 to disable streaming.
        prefetch_blocks: Number of blocks to prefetch ahead using async transfers.
            Higher values hide more latency but use more GPU memory.
        use_pinned_memory: Whether to use pinned (page-locked) memory for faster
            CPU<->GPU transfers. Requires sufficient system RAM.
        use_non_blocking: Whether to use non-blocking transfers. Faster but
            requires careful synchronization.
        offload_device: Device to offload blocks to (default: CPU).
        compute_device: Device to run computation on (default: CUDA).
        debug: Enable debug logging for transfer timing.
    """

    blocks_to_stream: int = 0
    prefetch_blocks: int = 1
    use_pinned_memory: bool = True
    use_non_blocking: bool = True
    offload_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    compute_device: torch.device | None = None  # Auto-detect if None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.compute_device is None:
            self.compute_device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )


class BlockStreamingState:
    """Manages state for block streaming during inference.

    This class tracks which blocks are currently on GPU, manages CUDA streams
    for async transfers, and coordinates prefetching.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        config: BlockStreamingConfig,
    ) -> None:
        self.blocks = blocks
        self.config = config
        self.num_blocks = len(blocks)

        # Determine which blocks to keep on GPU vs stream
        self.num_resident = max(0, self.num_blocks - config.blocks_to_stream)
        self.streaming_start_idx = self.num_resident

        # Track block locations
        self.block_on_gpu: list[bool] = [False] * self.num_blocks

        # CUDA streams for async transfers
        self._streams: list[torch.cuda.Stream] = []
        self._stream_idx = 0

        # Pinned memory buffers (lazily allocated)
        self._pinned_buffers: dict[str, torch.Tensor] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        # Statistics
        self.total_transfers = 0
        self.total_transfer_bytes = 0

        logger.info(
            f"BlockStreamingState initialized: {self.num_blocks} blocks, "
            f"{self.num_resident} resident on GPU, "
            f"{config.blocks_to_stream} streaming"
        )

    def _get_stream(self) -> torch.cuda.Stream | None:
        """Get a CUDA stream for async transfers."""
        if not torch.cuda.is_available():
            return None

        if not self._streams:
            # Create streams lazily
            num_streams = max(2, self.config.prefetch_blocks + 1)
            self._streams = [
                torch.cuda.Stream(device=self.config.compute_device)
                for _ in range(num_streams)
            ]
            logger.debug(f"Created {num_streams} CUDA streams for block streaming")

        stream = self._streams[self._stream_idx]
        self._stream_idx = (self._stream_idx + 1) % len(self._streams)
        return stream

    def _pin_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Pin a tensor to page-locked memory for faster transfers."""
        if not self.config.use_pinned_memory:
            return tensor

        if tensor.is_pinned():
            return tensor

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Try to pin the memory
        try:
            # Create a pinned copy
            pinned = torch.empty_like(tensor, pin_memory=True)
            pinned.copy_(tensor)
            return pinned
        except RuntimeError as e:
            logger.warning(f"Failed to pin tensor {name}: {e}")
            return tensor

    def move_block_to_gpu(
        self,
        block_idx: int,
        stream: torch.cuda.Stream | None = None,
        sync: bool = True,
    ) -> None:
        """Move a block's parameters to GPU.

        Args:
            block_idx: Index of the block to move.
            stream: CUDA stream for async transfer. If None, uses default stream.
            sync: Whether to synchronize after transfer.
        """
        if block_idx < 0 or block_idx >= self.num_blocks:
            return

        if self.block_on_gpu[block_idx]:
            return

        block = self.blocks[block_idx]
        device = self.config.compute_device
        non_blocking = self.config.use_non_blocking and stream is not None

        with self._lock:
            if self.config.debug:
                logger.debug(
                    f"Moving block {block_idx} to GPU (stream={stream is not None})"
                )

            # Use stream context if provided
            ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()

            with ctx:
                bytes_transferred = 0
                for param in block.parameters():
                    if param.device != device:
                        param.data = param.data.to(device, non_blocking=non_blocking)
                        bytes_transferred += param.numel() * param.element_size()

                for buffer in block.buffers():
                    if buffer.device != device:
                        buffer.data = buffer.data.to(device, non_blocking=non_blocking)
                        bytes_transferred += buffer.numel() * buffer.element_size()

                self.total_transfers += 1
                self.total_transfer_bytes += bytes_transferred

            if sync and stream is not None:
                stream.synchronize()

            self.block_on_gpu[block_idx] = True

    def move_block_to_cpu(
        self,
        block_idx: int,
        stream: torch.cuda.Stream | None = None,
        sync: bool = True,
    ) -> None:
        """Move a block's parameters to CPU.

        Args:
            block_idx: Index of the block to move.
            stream: CUDA stream for async transfer.
            sync: Whether to synchronize after transfer.
        """
        if block_idx < 0 or block_idx >= self.num_blocks:
            return

        if not self.block_on_gpu[block_idx]:
            return

        # Don't offload resident blocks
        if block_idx < self.streaming_start_idx:
            return

        block = self.blocks[block_idx]
        device = self.config.offload_device
        non_blocking = self.config.use_non_blocking

        with self._lock:
            if self.config.debug:
                logger.debug(f"Moving block {block_idx} to CPU")

            # Synchronize current stream before offloading
            if stream is not None and sync:
                stream.synchronize()

            for param in block.parameters():
                if param.device != device:
                    # Pin the CPU tensor for faster future transfers
                    cpu_data = param.data.to(device, non_blocking=non_blocking)
                    if self.config.use_pinned_memory:
                        cpu_data = self._pin_tensor(cpu_data, f"block_{block_idx}")
                    param.data = cpu_data

            for buffer in block.buffers():
                if buffer.device != device:
                    buffer.data = buffer.data.to(device, non_blocking=non_blocking)

            self.block_on_gpu[block_idx] = False

    def prefetch_block(self, block_idx: int) -> torch.cuda.Stream | None:
        """Prefetch a block to GPU asynchronously.

        Returns the stream used for prefetching, which should be synchronized
        before using the block.
        """
        if block_idx < 0 or block_idx >= self.num_blocks:
            return None

        if self.block_on_gpu[block_idx]:
            return None

        stream = self._get_stream()
        if stream is not None:
            # Wait for current computation to finish before starting transfer
            stream.wait_stream(torch.cuda.current_stream())

        self.move_block_to_gpu(block_idx, stream=stream, sync=False)
        return stream

    def cleanup(self) -> None:
        """Clean up resources."""
        # Synchronize all streams
        for stream in self._streams:
            stream.synchronize()

        self._streams.clear()
        self._pinned_buffers.clear()

        if self.config.debug:
            logger.info(
                f"Block streaming stats: {self.total_transfers} transfers, "
                f"{self.total_transfer_bytes / 1024**2:.2f} MB total"
            )


# Null context manager for when streams aren't available
class nullcontext:
    """Context manager that does nothing."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        pass


def _create_forward_hook(
    state: BlockStreamingState,
    block_idx: int,
) -> Callable:
    """Create a pre-forward hook that ensures the block is on GPU."""

    def hook(module: nn.Module, inputs: tuple) -> None:
        # Ensure this block is on GPU
        if not state.block_on_gpu[block_idx]:
            state.move_block_to_gpu(block_idx, sync=True)

        # Prefetch next blocks
        for i in range(1, state.config.prefetch_blocks + 1):
            next_idx = block_idx + i
            if next_idx < state.num_blocks and not state.block_on_gpu[next_idx]:
                state.prefetch_block(next_idx)

    return hook


def _create_post_forward_hook(
    state: BlockStreamingState,
    block_idx: int,
) -> Callable:
    """Create a post-forward hook that offloads the block to CPU."""

    def hook(module: nn.Module, inputs: tuple, outputs: Any) -> None:
        # Only offload streaming blocks (not resident blocks)
        if block_idx >= state.streaming_start_idx:
            # Don't offload if we're within the prefetch window of a future block
            # This is handled by the prefetch logic
            state.move_block_to_cpu(block_idx, sync=False)

    return hook


def setup_block_streaming(
    blocks: nn.ModuleList,
    config: BlockStreamingConfig,
) -> BlockStreamingState:
    """Set up block streaming for a ModuleList of transformer blocks.

    This function:
    1. Moves streaming blocks to CPU (keeping resident blocks on GPU)
    2. Pins CPU memory for faster transfers
    3. Registers forward hooks for automatic load/offload

    Args:
        blocks: ModuleList of transformer blocks.
        config: Streaming configuration.

    Returns:
        BlockStreamingState that manages the streaming.
    """
    if config.blocks_to_stream <= 0:
        logger.info("Block streaming disabled (blocks_to_stream=0)")
        return BlockStreamingState(blocks, config)

    num_blocks = len(blocks)
    if config.blocks_to_stream >= num_blocks:
        logger.warning(
            f"blocks_to_stream ({config.blocks_to_stream}) >= num_blocks ({num_blocks}), "
            f"clamping to {num_blocks - 1}"
        )
        config.blocks_to_stream = num_blocks - 1

    state = BlockStreamingState(blocks, config)

    # Initial setup: move streaming blocks to CPU, keep resident blocks on GPU
    logger.info(
        f"Setting up block streaming: {state.num_resident} blocks on GPU, "
        f"{config.blocks_to_stream} blocks streaming from CPU"
    )

    # Synchronize before moving blocks
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Move streaming blocks to CPU
    for i in range(state.streaming_start_idx, num_blocks):
        block = blocks[i]
        for param in block.parameters():
            if param.device.type != "cpu":
                cpu_data = param.data.to(config.offload_device)
                if config.use_pinned_memory:
                    cpu_data = state._pin_tensor(cpu_data, f"block_{i}")
                param.data = cpu_data

        for buffer in block.buffers():
            if buffer.device.type != "cpu":
                buffer.data = buffer.data.to(config.offload_device)

        state.block_on_gpu[i] = False

    # Mark resident blocks as on GPU
    for i in range(state.streaming_start_idx):
        state.block_on_gpu[i] = True

    # Register hooks for automatic streaming
    for i in range(num_blocks):
        block = blocks[i]

        # Pre-forward hook: ensure block is on GPU, prefetch next blocks
        pre_hook = _create_forward_hook(state, i)
        block.register_forward_pre_hook(pre_hook)

        # Post-forward hook: offload block to CPU
        post_hook = _create_post_forward_hook(state, i)
        block.register_forward_hook(post_hook)

    # Force garbage collection to free any lingering GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Log memory after setup
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"After block streaming setup: {mem_allocated:.2f} GB allocated, "
            f"{mem_reserved:.2f} GB reserved"
        )

    return state


def cleanup_block_streaming(
    blocks: nn.ModuleList,
    state: BlockStreamingState | None = None,
    move_all_to_gpu: bool = False,
) -> None:
    """Clean up block streaming hooks and optionally move all blocks to GPU.

    Args:
        blocks: ModuleList of transformer blocks.
        state: BlockStreamingState to clean up.
        move_all_to_gpu: If True, move all blocks back to GPU after cleanup.
    """
    # Remove hooks
    for block in blocks:
        block._forward_pre_hooks.clear()
        block._forward_hooks.clear()

    if state is not None:
        state.cleanup()

    if move_all_to_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        for block in blocks:
            block.to(device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_block_memory(block: nn.Module) -> int:
    """Estimate memory usage of a transformer block in bytes."""
    total_bytes = 0
    for param in block.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in block.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes


def calculate_optimal_streaming_config(
    blocks: nn.ModuleList,
    available_vram_gb: float,
    safety_margin_gb: float = 2.0,
    min_resident_blocks: int = 4,
) -> BlockStreamingConfig:
    """Calculate optimal streaming configuration based on available VRAM.

    Args:
        blocks: ModuleList of transformer blocks.
        available_vram_gb: Available VRAM in GB.
        safety_margin_gb: Safety margin to leave free.
        min_resident_blocks: Minimum blocks to keep on GPU.

    Returns:
        Optimal BlockStreamingConfig.
    """
    num_blocks = len(blocks)
    if num_blocks == 0:
        return BlockStreamingConfig(blocks_to_stream=0)

    # Estimate memory per block
    block_memory = estimate_block_memory(blocks[0])
    block_memory_gb = block_memory / 1024**3

    # Calculate how many blocks can fit
    usable_vram = available_vram_gb - safety_margin_gb
    max_resident = int(usable_vram / block_memory_gb)
    max_resident = max(min_resident_blocks, min(max_resident, num_blocks))

    blocks_to_stream = num_blocks - max_resident

    logger.info(
        f"Optimal streaming config: {block_memory_gb:.2f} GB/block, "
        f"{max_resident} resident, {blocks_to_stream} streaming"
    )

    return BlockStreamingConfig(
        blocks_to_stream=blocks_to_stream,
        prefetch_blocks=1,
        use_pinned_memory=True,
        use_non_blocking=True,
    )
