from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


def initialize_kv_bank(
    generator,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    bank_size: int,
    frame_seq_length: int,
    kv_bank_existing: list[dict] | None = None,
    reset_indices: bool = True,
):
    """
    Initialize a KV bank for the memory bank mechanism.

    The KV bank stores compressed historical context using a memory bank approach.
    Each block has storage for:
    - k, v: The main bank storage (bank_size * frame_seq_length tokens)
    - k_new, v_new: Temporary storage for new incoming tokens (frame_seq_length tokens)
    - Indices for tracking the current position in the bank

    Args:
        generator: The transformer model
        batch_size: Batch size for the cache
        dtype: Data type for the cache tensors
        device: Device to create the cache on
        bank_size: Number of frames to store in the bank
        frame_seq_length: Number of tokens per frame
        kv_bank_existing: Existing KV bank to reuse if shapes match
        reset_indices: Whether to reset the end indices to 0

    Returns:
        List of dictionaries, one per transformer block, containing:
        - k, v: Main bank storage tensors
        - k_new, v_new: New token storage tensors
        - global_end_index, local_end_index: Position tracking
    """
    kv_bank = []

    # Calculate bank size in tokens
    kv_bank_size = bank_size * frame_seq_length

    # Get transformer config
    num_transformer_blocks = len(generator.model.blocks)
    num_heads = generator.model.num_heads
    dim = generator.model.dim
    head_dim = dim // num_heads

    # Define tensor shapes
    k_shape = [batch_size, kv_bank_size, num_heads, head_dim]
    v_shape = [batch_size, kv_bank_size, num_heads, head_dim]
    k_new_shape = [batch_size, frame_seq_length, num_heads, head_dim]
    v_new_shape = [batch_size, frame_seq_length, num_heads, head_dim]

    # Check if we can reuse existing bank
    if (
        kv_bank_existing
        and len(kv_bank_existing) > 0
        and list(kv_bank_existing[0]["k"].shape) == k_shape
        and list(kv_bank_existing[0]["v"].shape) == v_shape
        and list(kv_bank_existing[0]["k_new"].shape) == k_new_shape
        and list(kv_bank_existing[0]["v_new"].shape) == v_new_shape
    ):
        # Reuse existing bank by zeroing it out
        for i in range(num_transformer_blocks):
            kv_bank_existing[i]["k"].zero_()
            kv_bank_existing[i]["v"].zero_()
            kv_bank_existing[i]["k_new"].zero_()
            kv_bank_existing[i]["v_new"].zero_()

            if reset_indices:
                kv_bank_existing[i]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )
                kv_bank_existing[i]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=device
                )

        return kv_bank_existing
    else:
        # Create new bank
        for _ in range(num_transformer_blocks):
            kv_bank.append(
                {
                    "k": torch.zeros(k_shape, dtype=dtype, device=device).contiguous(),
                    "v": torch.zeros(v_shape, dtype=dtype, device=device).contiguous(),
                    "k_new": torch.zeros(
                        k_new_shape, dtype=dtype, device=device
                    ).contiguous(),
                    "v_new": torch.zeros(
                        v_new_shape, dtype=dtype, device=device
                    ).contiguous(),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )
        return kv_bank


class SetupMemoryBankBlock(ModularPipelineBlocks):
    model_name = "MemFlow"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("bank_size", 3),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return "Setup Memory Bank block initializes a KV memory bank to be used during generation"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "kv_bank",
                type_hint=list[dict] | None,
                default=None,
                description="Existing KV memory bank",
            ),
            InputParam(
                "init_cache",
                type_hint=bool,
                default=False,
                description="Whether to (re)initialize memory bank",
            ),
            InputParam(
                "height",
                required=True,
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                required=True,
                type_hint=int,
                description="Width of the video",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "kv_bank",
                type_hint=list[dict],
                description="Initialized KV memory bank",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        init_cache = block_state.init_cache

        # Calculate frame sequence length from video dimensions
        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        # Get generator parameters
        generator_param = next(components.generator.model.parameters())
        generator_dtype = generator_param.dtype
        generator_device = generator_param.device

        # Initialize or reinitialize the memory bank if needed
        if init_cache or block_state.kv_bank is None:
            block_state.kv_bank = initialize_kv_bank(
                generator=components.generator,
                batch_size=1,
                dtype=generator_dtype,
                device=generator_device,
                bank_size=components.config.bank_size,
                frame_seq_length=frame_seq_length,
                kv_bank_existing=block_state.kv_bank,
            )

        self.set_block_state(state, block_state)
        return components, state
