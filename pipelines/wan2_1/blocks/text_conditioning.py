import logging
from typing import Any

import torch
from diffusers.modular_pipelines import BlockState, ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from ...blending import parse_transition_config

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
                description="New prompts to condition denoising (list[dict] format handled by EmbeddingBlendingBlock)",
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Text embeddings to condition denoising",
            ),
            InputParam(
                "transition",
                type_hint=dict | None,
                description="Optional transition config containing target prompts to encode",
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
            OutputParam(
                "prompt_embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of individual embeddings for blending (when prompts is list[dict])",
            ),
            OutputParam(
                "prompt_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to prompt_embeds_list",
            ),
            OutputParam(
                "target_embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of pre-encoded transition target embeddings",
            ),
            OutputParam(
                "target_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to target_embeds_list",
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
        block_state.prompt_embeds_list = None
        block_state.prompt_weights = None
        block_state.target_embeds_list = None
        block_state.target_weights = None

        # Check if prompts changed
        prompts_changed = (
            block_state.current_prompts is None
            or block_state.current_prompts != block_state.prompts
        )

        # Diagnostic logging
        logger.info(
            f"TextConditioningBlock: current_prompts={block_state.current_prompts}, "
            f"prompts={block_state.prompts}, prompts_changed={prompts_changed}, "
            f"transition={block_state.transition is not None}"
        )

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):
            # Encode regular prompts if they changed
            if prompts_changed:
                # Handle list[dict] format (for blending)
                if (
                    isinstance(block_state.prompts, list)
                    and len(block_state.prompts) > 0
                    and isinstance(block_state.prompts[0], dict)
                ):
                    # Encode each prompt individually and extract weights
                    embeddings_list = []
                    weights_list = []

                    for prompt_item in block_state.prompts:
                        text = prompt_item.get("text", "")
                        weight = prompt_item.get("weight", 1.0)

                        # Encode individual prompt
                        conditional_dict = components.text_encoder(text_prompts=[text])
                        embeddings_list.append(conditional_dict["prompt_embeds"])
                        weights_list.append(weight)

                    # Store list of embeddings and weights for EmbeddingBlendingBlock
                    block_state.prompt_embeds_list = embeddings_list
                    block_state.prompt_weights = weights_list
                    # Don't set prompt_embeds here - EmbeddingBlendingBlock will blend and set it

                # Handle simple str or list[str] format (backward compatibility)
                else:
                    conditional_dict = components.text_encoder(
                        text_prompts=[block_state.prompts]
                        if isinstance(block_state.prompts, str)
                        else block_state.prompts
                    )
                    block_state.prompt_embeds = conditional_dict["prompt_embeds"]

                block_state.current_prompts = block_state.prompts
                block_state.prompt_embeds_updated = True

            # Handle transition target encoding (independent of prompts_changed)
            if block_state.transition is not None:
                target_prompts, _, _, _ = parse_transition_config(
                    block_state.transition
                )

                if target_prompts:
                    # Encode each target prompt individually
                    target_embeddings = []
                    target_weights_list = []

                    for prompt_item in target_prompts:
                        text = prompt_item.get("text", "")
                        weight = prompt_item.get("weight", 1.0)

                        # Encode target prompt
                        conditional_dict = components.text_encoder(text_prompts=[text])
                        target_embeddings.append(conditional_dict["prompt_embeds"])
                        target_weights_list.append(weight)

                    # Store target embeddings and weights for EmbeddingBlendingBlock
                    block_state.target_embeds_list = target_embeddings
                    block_state.target_weights = target_weights_list

        self.set_block_state(state, block_state)
        return components, state
