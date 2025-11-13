import logging
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from ...blending import EmbeddingBlender, parse_transition_config

logger = logging.getLogger(__name__)


class EmbeddingBlendingBlock(ModularPipelineBlocks):
    """Embedding Blending block that handles spatial and temporal embedding blending.

    This block orchestrates the EmbeddingBlender component within the modular pipeline architecture.

    Responsibilities:
    - Spatial blending: Combining multiple weighted embeddings into a single embedding
    - Temporal blending: Smooth transitions between embeddings over multiple frames
    - Cache management: Setting prompt_embeds_updated flag for downstream cache reinitialization
    - Dtype conversion: Ensuring embeddings match pipeline dtype (e.g., bfloat16)
    - State management: Integrating EmbeddingBlender state with pipeline state flow

    Architecture Notes:
    - This block is a thin integration layer around the EmbeddingBlender business logic class
    - EmbeddingBlender remains separate for testability and separation of concerns
    - During transitions, we set prompt_embeds_updated=True to reset ONLY cross-attention cache,
      preserving KV cache for smooth temporal coherence (unlike init_cache which resets everything)

    Cache Reset Strategy:
    - prompt_embeds_updated=True → Resets cross-attn cache only (SetupCachesBlock)
    - init_cache=True → Resets ALL caches (KV, cross-attn, VAE, frame counter)
    - We use prompt_embeds_updated during transitions to maintain temporal context
    """

    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("embedding_blender", EmbeddingBlender),
            ComponentSpec(
                "text_encoder", torch.nn.Module
            ),  # Only for transition target encoding
        ]

    @property
    def description(self) -> str:
        return "Embedding Blending block that handles spatial and temporal embedding blending"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt_embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of pre-encoded embeddings to blend",
            ),
            InputParam(
                "prompt_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to prompt_embeds_list",
            ),
            InputParam(
                "prompt_interpolation_method",
                type_hint=str,
                default="linear",
                description="Spatial interpolation method for blending: 'linear' or 'slerp'",
            ),
            InputParam(
                "transition",
                type_hint=dict | None,
                description="Optional transition config for temporal blending",
            ),
            InputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Existing prompt embeddings (optional)",
            ),
            InputParam(
                "prompts",
                type_hint=list[dict] | None,
                description="Prompts for transition encoding (list[dict] format)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Blended text embeddings to condition denoising",
            ),
            OutputParam(
                "prompt_embeds_updated",
                type_hint=bool,
                description="Whether text embeddings were updated (requires cross-attention cache re-initialization)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        # Get inputs from state
        prompt_embeds_list = block_state.prompt_embeds_list
        prompt_weights = block_state.prompt_weights
        prompt_interpolation_method = (
            block_state.prompt_interpolation_method or "linear"
        )
        transition = block_state.transition

        logger.info(
            f"__call__: Starting with prompt_embeds_list={prompt_embeds_list is not None}, "
            f"transition={transition is not None}, is_transitioning={components.embedding_blender.is_transitioning()}"
        )

        # Initialize flag
        block_state.prompt_embeds_updated = False

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):
            # Step 1: Spatial blending - blend pre-encoded embeddings if available
            if prompt_embeds_list and prompt_weights:
                logger.info(
                    f"__call__: Blending {len(prompt_embeds_list)} pre-encoded embeddings"
                )

                blended_embeds = components.embedding_blender.blend(
                    embeddings=prompt_embeds_list,
                    weights=prompt_weights,
                    interpolation_method=prompt_interpolation_method,
                )

                if blended_embeds is not None:
                    block_state.prompt_embeds = blended_embeds.to(
                        dtype=components.config.dtype
                    )
                    block_state.prompt_embeds_updated = True
                    logger.info("__call__: Successfully blended embeddings")

            # Step 2: Handle transition requests (temporal blending)
            # Only process transition if we're NOT already transitioning to prevent restart loops
            if (
                transition is not None
                and not components.embedding_blender.is_transitioning()
            ):
                logger.info("__call__: Processing transition request")

                target_prompts, num_steps, temporal_method, is_immediate = (
                    parse_transition_config(transition)
                )

                if target_prompts:
                    # Encode and blend target prompts
                    logger.info(
                        f"__call__: Encoding {len(target_prompts)} target prompts"
                    )

                    target_embeddings = []
                    target_weights = []

                    for prompt_item in target_prompts:
                        text = prompt_item.get("text", "")
                        weight = prompt_item.get("weight", 1.0)

                        # Encode target prompt
                        conditional_dict = components.text_encoder(text_prompts=[text])
                        target_embeddings.append(conditional_dict["prompt_embeds"])
                        target_weights.append(weight)

                    # Blend target embeddings (don't cache to preserve current blend for comparison)
                    target_blend = components.embedding_blender.blend(
                        embeddings=target_embeddings,
                        weights=target_weights,
                        interpolation_method=prompt_interpolation_method,
                        cache_result=False,
                    )

                    if target_blend is not None:
                        if is_immediate:
                            # Immediate transition (num_steps=0)
                            logger.info("__call__: Applying immediate transition")
                            block_state.prompt_embeds = target_blend.to(
                                dtype=components.config.dtype
                            )
                            block_state.prompt_embeds_updated = True
                        else:
                            # Smooth transition (num_steps>0)
                            logger.info(
                                f"__call__: Starting smooth transition over {num_steps} steps"
                            )
                            components.embedding_blender.start_transition(
                                target_embedding=target_blend,
                                num_steps=num_steps,
                                temporal_interpolation_method=temporal_method,
                            )

            elif (
                transition is not None
                and components.embedding_blender.is_transitioning()
            ):
                logger.info(
                    "__call__: Ignoring transition request - already transitioning"
                )

            # Step 3: Get next embedding from transition queue (if transitioning)
            if components.embedding_blender.is_transitioning():
                logger.info("__call__: Getting next embedding from transition queue")
                is_transitioning_before = True
                next_embedding = components.embedding_blender.get_next_embedding()
                is_transitioning_after = components.embedding_blender.is_transitioning()

                if next_embedding is not None:
                    # Cast to pipeline dtype before storing
                    next_embedding = next_embedding.to(dtype=components.config.dtype)
                    block_state.prompt_embeds = next_embedding
                    block_state.prompt_embeds_updated = True
                    logger.info(
                        f"__call__: Updated prompt_embeds from transition "
                        f"(still transitioning: {is_transitioning_after})"
                    )

        logger.info(
            f"__call__: Finished, prompt_embeds_updated={block_state.prompt_embeds_updated}, "
            f"is_transitioning={components.embedding_blender.is_transitioning()}"
        )

        self.set_block_state(state, block_state)
        return components, state
