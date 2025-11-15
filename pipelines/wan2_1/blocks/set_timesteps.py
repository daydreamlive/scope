from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)


class SetTimestepsBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("scheduler", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Set Timesteps block that configures denoising steps based on scheduler"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "denoising_step_list",
                required=True,
                type_hint=list[int] | torch.Tensor,
                description="List of denoising steps",
            ),
            InputParam(
                "manage_cache",
                default=True,
                type_hint=bool,
                description="Whether to automatically determine to (re)initialize caches",
            ),
            # The following should be converted to intermediate inputs to denote that they can come from other blocks
            # and can be modified since they are also listed under intermediate outputs. They are included as inputs for now
            # because of what seems to be a bug where intermediate inputs cannot be simplify accessed in block state via
            # block_state.<intermediate_input>
            InputParam(
                "current_denoising_step_list",
                type_hint=torch.Tensor | None,
                description="Current list of denoising steps",
            ),
            InputParam(
                "init_cache",
                default=False,
                type_hint=bool,
                description="Whether to (re)initialize caches",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "current_denoising_step_list",
                type_hint=torch.Tensor,
                description="Current list of denoising steps",
            ),
            OutputParam(
                "init_cache",
                type_hint=bool,
                description="Whether to (re)initialize caches",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        denoising_step_list = block_state.denoising_step_list
        if isinstance(block_state.denoising_step_list, list):
            denoising_step_list = torch.tensor(
                denoising_step_list,
                dtype=torch.long,
            )

        if block_state.current_denoising_step_list is None or not torch.equal(
            block_state.current_denoising_step_list, denoising_step_list
        ):
            block_state.current_denoising_step_list = denoising_step_list.clone()

            if block_state.manage_cache:
                block_state.init_cache = True

        self.set_block_state(state, block_state)
        return components, state
