import logging
from typing import Any

import torch
from diffusers.modular_pipelines import BlockState, ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

logger = logging.getLogger(__name__)


class TextConditioningBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return (
            "Text Conditioning block that encodes prompts into embeddings for downstream blending. "
            "Transition configuration is forwarded via pipeline state and handled by the "
            "embedding blending block."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "current_prompts",
                type_hint=list[dict] | None,
                description="Current prompts conditioning denoising",
            ),
            InputParam(
                "prompts",
                required=True,
                type_hint=list[dict],
                description="Prompts to condition denoising as list of dicts with 'text' and 'weight' keys",
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Text embeddings to condition denoising",
            ),
            InputParam(
                "transition",
                type_hint=dict | None,
                description=(
                    "Optional transition config for temporal blending. This block does not interpret "
                    "the transition directly; it is consumed by downstream blending logic."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "current_prompts",
                type_hint=list[dict],
                description="Current prompts conditioning denoising",
            ),
            OutputParam(
                "embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of individual embeddings for blending (when prompts is list[dict])",
            ),
            OutputParam(
                "embedding_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to embeds_list",
            ),
            OutputParam(
                "conditioning_changed",
                type_hint=bool,
                description=(
                    "Whether conditioning inputs changed since last call. Used by "
                    "downstream blocks (e.g. embedding blending) to manage transitions."
                ),
            ),
        ]

    @staticmethod
    def check_inputs(block_state: BlockState):
        if block_state.prompts is not None:
            if not isinstance(block_state.prompts, list):
                raise ValueError(
                    f"`prompts` must be a list[dict] but is {type(block_state.prompts)}"
                )
            if len(block_state.prompts) > 0 and not isinstance(
                block_state.prompts[0], dict
            ):
                raise ValueError(
                    f"`prompts` must be a list[dict] with 'text' and 'weight' keys, "
                    f"but first element is {type(block_state.prompts[0])}"
                )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        # Initialize outputs
        block_state.embeds_list = None
        block_state.embedding_weights = None

        # Check if prompts changed
        prompts_changed = (
            block_state.current_prompts is None
            or block_state.current_prompts != block_state.prompts
        )
        block_state.conditioning_changed = prompts_changed

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):

            def encode_prompt_items(
                prompt_items: list[dict],
            ) -> tuple[list[torch.Tensor], list[float]]:
                """Encode a list of prompt dicts into embeddings and weights."""
                embeddings: list[torch.Tensor] = []
                weights: list[float] = []

                for prompt_item in prompt_items:
                    text = prompt_item.get("text", "")
                    weight = prompt_item.get("weight", 1.0)

                    conditional_dict = components.text_encoder(text_prompts=[text])
                    embeddings.append(conditional_dict["prompt_embeds"])
                    weights.append(weight)

                return embeddings, weights

            # Encode regular prompts if they changed
            if prompts_changed:
                (
                    block_state.embeds_list,
                    block_state.embedding_weights,
                ) = encode_prompt_items(block_state.prompts)

                block_state.current_prompts = block_state.prompts

        self.set_block_state(state, block_state)
        return components, state
