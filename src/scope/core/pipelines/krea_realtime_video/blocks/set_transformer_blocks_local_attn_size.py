from typing import Any

import torch
from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class SetTransformerBlocksLocalAttnSizeBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("local_attn_size", 6),
        ]

    @property
    def description(self) -> str:
        return (
            "Set Transformer Blocks Local Attn Size block sets the local_attn_size "
            "of transformer blocks to the config value"
        )

    @property
    def inputs(self) -> list[InputParam]:
        return []

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        for block in components.generator.model.blocks:
            block.self_attn.local_attn_size = components.config.local_attn_size

        self.set_block_state(state, block_state)
        return components, state


class EmptyBlock(ModularPipelineBlocks):
    """Empty block that does nothing. Used as a no-op placeholder in routing."""

    def __call__(self, components, state):
        return components, state


class AutoSetTransformerBlocksLocalAttnSizeBlock(AutoPipelineBlocks):
    """Auto-routing block for setting transformer local attention size.

    Routes to SetTransformerBlocksLocalAttnSizeBlock when 'video' input is provided
    (V2V mode), otherwise does nothing. This ensures local_attn_size is only set
    for video-to-video mode, preserving T2V mode behavior.
    """

    block_classes = [
        SetTransformerBlocksLocalAttnSizeBlock,
        EmptyBlock,
    ]

    block_names = [
        "set_transformer_blocks_local_attn_size",
        "no_set_local_attn_size",
    ]

    block_trigger_inputs = [
        "video",
        None,
    ]

    @property
    def description(self):
        return (
            "AutoSetTransformerBlocksLocalAttnSizeBlock: Routes local attention size setting:\n"
            " - Sets local_attn_size when 'video' input is provided (V2V mode)\n"
            " - Skips setting when no 'video' input is provided (T2V mode)\n"
        )
