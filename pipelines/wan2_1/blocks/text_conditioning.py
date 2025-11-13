from typing import Any

import torch
from diffusers.modular_pipelines import BlockState, ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)


class TextConditioningBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Text Conditioning block that generates text embeddings to condition denoising"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "current_prompts",
                type_hint=str | list[str] | list[dict] | None,
                description="Current prompts conditioning denoising",
            ),
            InputParam(
                "prompts",
                required=True,
                type_hint=str | list[str] | list[dict],
                description="New prompts to condition denoising (list[dict] format handled by PromptBlendingBlock)",
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Text embeddings to condition denoising",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "current_prompts",
                type_hint=str | list[str] | list[dict],
                description="Current prompts conditioning denoising",
            ),
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Text embeddings to condition denoising",
            ),
            OutputParam(
                "prompt_embeds_updated",
                type_hint=bool,
                description="Whether text embeddings were updated (requires cross-attention cache re-initialization)",
            ),
        ]

    @staticmethod
    def check_inputs(block_state: BlockState):
        if (
            block_state.prompts is not None
            and not isinstance(block_state.prompts, str)
            and not isinstance(block_state.prompts, list)
        ):
            raise ValueError(
                f"`prompts` has to be of type `str` or `list` but is {type(block_state.prompts)}"
            )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        block_state.prompt_embeds_updated = False

        # Skip if prompts are list[dict] - PromptBlendingBlock handles these
        # Note: We check first element to distinguish list[dict] from list[str]
        # API always sends list[dict] (PromptItem), but blocks support list[str] for internal use
        if (
            isinstance(block_state.prompts, list)
            and len(block_state.prompts) > 0
            and isinstance(block_state.prompts[0], dict)
        ):
            self.set_block_state(state, block_state)
            return components, state

        # Only run text_encoder if prompt changed
        if (
            block_state.current_prompts is None
            or block_state.current_prompts != block_state.prompts
        ):
            with torch.autocast(
                str(components.config.device), dtype=components.config.dtype
            ):
                conditional_dict = components.text_encoder(
                    text_prompts=[block_state.prompts]
                    if isinstance(block_state.prompts, str)
                    else block_state.prompts
                )
            block_state.prompt_embeds = conditional_dict["prompt_embeds"]
            block_state.current_prompts = block_state.prompts
            block_state.prompt_embeds_updated = True

        self.set_block_state(state, block_state)
        return components, state
