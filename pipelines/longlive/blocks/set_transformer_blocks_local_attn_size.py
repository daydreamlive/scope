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
        return "Set Transformer Blocks Local Attn Size block sets the local_attn_size of transformer blocks to the config value"

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
