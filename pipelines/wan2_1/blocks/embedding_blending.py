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
    Currently used for text prompt embeddings, but the blending logic itself is generic.

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
        ]

    @property
    def description(self) -> str:
        return "Embedding Blending block that handles spatial and temporal embedding blending"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of pre-encoded embeddings to blend",
            ),
            InputParam(
                "embedding_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to embeds_list",
            ),
            InputParam(
                "spatial_interpolation_method",
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
                "target_embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of pre-encoded transition target embeddings",
            ),
            InputParam(
                "target_weights",
                type_hint=list[float] | None,
                description="List of weights corresponding to target_embeds_list",
            ),
            InputParam(
                "conditioning_changed",
                type_hint=bool,
                default=False,
                description=(
                    "Whether conditioning inputs changed since last call. Used to "
                    "manage transitions when new conditioning arrives."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "prompt_embeds",
                type_hint=torch.Tensor,
                description="Blended embeddings to condition denoising (pipeline state variable)",
            ),
            OutputParam(
                "prompt_embeds_updated",
                type_hint=bool,
                description="Whether embeddings were updated (requires cross-attention cache re-initialization)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        # Get inputs from state
        embeds_list = block_state.embeds_list
        embedding_weights = block_state.embedding_weights
        spatial_interpolation_method = (
            block_state.spatial_interpolation_method or "linear"
        )
        transition = block_state.transition
        target_embeds_list = block_state.target_embeds_list
        target_weights = block_state.target_weights
        conditioning_changed = getattr(block_state, "conditioning_changed", False)

        # Initialize flag
        block_state.prompt_embeds_updated = False

        if conditioning_changed and components.embedding_blender.is_transitioning():
            logger.info(
                "EmbeddingBlendingBlock: Conditioning changed during transition, "
                "cancelling transition"
            )
            components.embedding_blender.cancel_transition()

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):
            # Step 1: Spatial blending - blend pre-encoded embeddings if available
            if embeds_list and embedding_weights:
                blended_embeds = components.embedding_blender.blend(
                    embeddings=embeds_list,
                    weights=embedding_weights,
                    interpolation_method=spatial_interpolation_method,
                )

                if blended_embeds is not None:
                    block_state.prompt_embeds = blended_embeds.to(
                        dtype=components.config.dtype
                    )
                    block_state.prompt_embeds_updated = True

            # Step 2: Handle transition requests (temporal blending)
            if transition is not None:
                _, num_steps, temporal_method, is_immediate = parse_transition_config(
                    transition
                )

                # Use pre-encoded target embeddings from TextConditioningBlock
                if target_embeds_list and target_weights:
                    # Blend target embeddings (don't cache to preserve current blend for comparison)
                    target_blend = components.embedding_blender.blend(
                        embeddings=target_embeds_list,
                        weights=target_weights,
                        interpolation_method=spatial_interpolation_method,
                        cache_result=False,
                    )

                    if target_blend is not None:
                        if is_immediate:
                            # Immediate transition (num_steps=0)
                            block_state.prompt_embeds = target_blend.to(
                                dtype=components.config.dtype
                            )
                            block_state.prompt_embeds_updated = True
                        else:
                            # Smooth transition (num_steps>0)
                            components.embedding_blender.start_transition(
                                target_embedding=target_blend,
                                num_steps=num_steps,
                                temporal_interpolation_method=temporal_method,
                            )

            # Step 3: Get next embedding from transition queue (if transitioning)
            if components.embedding_blender.is_transitioning():
                next_embedding = components.embedding_blender.get_next_embedding()

                if next_embedding is not None:
                    # Cast to pipeline dtype before storing
                    next_embedding = next_embedding.to(dtype=components.config.dtype)
                    block_state.prompt_embeds = next_embedding
                    block_state.prompt_embeds_updated = True

        self.set_block_state(state, block_state)
        return components, state
