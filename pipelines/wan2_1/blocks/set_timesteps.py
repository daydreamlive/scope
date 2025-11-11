from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
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
                "current_denoising_step_list",
                type_hint=torch.Tensor | None,
                description="Current list of denoising steps",
            ),
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
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            InputParam(
                "current_denoising_step_list",
                type_hint=torch.Tensor,
                description="Current list of denoising steps",
            ),
            OutputParam(
                "denoising_step_list",
                type_hint=torch.Tensor,
                description="List of denoising steps",
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
        if not isinstance(denoising_step_list, torch.Tensor):
            denoising_step_list = torch.tensor(
                denoising_step_list,
                dtype=torch.long,
            )

        block_state.init_cache = False

        # Use torch.allclose() because we might have to compare a long vs. float32 tensor
        if block_state.current_denoising_step_list is None or not torch.allclose(
            block_state.current_denoising_step_list, denoising_step_list, atol=1e-1
        ):
            block_state.denoising_step_list = denoising_step_list

            if components.config.warp_denoising_step:
                # Add 0 as the final step
                timesteps = torch.cat(
                    (
                        components.scheduler.timesteps.cpu(),
                        torch.tensor([0], dtype=torch.float32),
                    )
                )
                # Reverse direction
                block_state.denoising_step_list = timesteps[
                    1000 - block_state.denoising_step_list
                ]

            block_state.current_denoising_step_list = block_state.denoising_step_list

            if block_state.manage_cache:
                block_state.init_cache = True

        self.set_block_state(state, block_state)
        return components, state
