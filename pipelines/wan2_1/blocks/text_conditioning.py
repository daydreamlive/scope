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
                type_hint=str | list[str] | list[dict] | None,
                description="Current prompts conditioning denoising",
            ),
            InputParam(
                "prompts",
                required=True,
                type_hint=str | list[str] | list[dict],
                description=(
                    "Prompts to condition denoising. Can be a string, list of strings, "
                    "or list of dicts with 'text' and 'weight' keys. Strings are treated as weight 1.0."
                ),
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
                type_hint=str | list[str] | list[dict],
                description="Current prompts conditioning denoising",
            ),
            OutputParam(
                "embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of individual embeddings for blending (when prompts is list[dict])",
            ),
            OutputParam(
                "embeds_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to embeds_list",
            ),
        ]

    @staticmethod
    def _normalize_prompts(prompts: str | list[str] | list[dict]) -> list[dict]:
        """Normalize prompts to list[dict] format."""
        if isinstance(prompts, str):
            return [{"text": prompts, "weight": 1.0}]
        if isinstance(prompts, list):
            if len(prompts) == 0:
                return []
            # Check if it's a list of strings
            if isinstance(prompts[0], str):
                return [{"text": text, "weight": 1.0} for text in prompts]
            # Otherwise assume it's already list[dict]
            return prompts
        raise ValueError(
            f"`prompts` must be str, list[str], or list[dict] but is {type(prompts)}"
        )

    @staticmethod
    def check_inputs(block_state: BlockState):
        if block_state.prompts is not None:
            if not isinstance(block_state.prompts, str | list):
                raise ValueError(
                    f"`prompts` must be str, list[str], or list[dict] but is {type(block_state.prompts)}"
                )
            if isinstance(block_state.prompts, list) and len(block_state.prompts) > 0:
                first_item = block_state.prompts[0]
                if not isinstance(first_item, str | dict):
                    raise ValueError(
                        f"`prompts` list must contain str or dict elements, "
                        f"but first element is {type(first_item)}"
                    )

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        # Initialize outputs
        block_state.embeds_list = None
        block_state.embeds_weights = None

        # Normalize prompts to list[dict] format for processing
        normalized_prompts = self._normalize_prompts(block_state.prompts)

        # Check if prompts changed (compare normalized versions)
        prompts_changed = (
            block_state.current_prompts is None
            or self._normalize_prompts(block_state.current_prompts)
            != normalized_prompts
        )

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):

            def encode_prompt_items(
                prompt_items: list[dict],
            ) -> tuple[list[torch.Tensor], list[float]]:
                """Encode a list of prompt dicts into embeddings and weights."""
                if not prompt_items:
                    return [], []

                # Extract texts and weights
                texts = [item.get("text", "") for item in prompt_items]
                weights = [item.get("weight", 1.0) for item in prompt_items]

                # Batch encode all prompts at once
                conditional_dict = components.text_encoder(text_prompts=texts)
                batched_embeds = conditional_dict["prompt_embeds"]

                # Split batched tensor into individual tensors, preserving batch dimension
                # batched_embeds shape: [batch_size, seq_len, hidden_dim]
                # Each embedding should be [1, seq_len, hidden_dim] to match original behavior
                embeddings = [
                    batched_embeds[i].unsqueeze(0)
                    for i in range(batched_embeds.shape[0])
                ]

                return embeddings, weights

            # Encode regular prompts if they changed
            if prompts_changed:
                (
                    block_state.embeds_list,
                    block_state.embeds_weights,
                ) = encode_prompt_items(normalized_prompts)

                # Store original prompts format (not normalized) for comparison
                block_state.current_prompts = block_state.prompts

        self.set_block_state(state, block_state)
        return components, state
