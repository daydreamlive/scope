import logging
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from scope.core.pipelines.blending import EmbeddingBlender, parse_transition_config

logger = logging.getLogger(__name__)


class EmbeddingBlendingBlock(ModularPipelineBlocks):
    """Embedding Blending block that handles spatial and temporal embedding blending.

    This block orchestrates the EmbeddingBlender component within the modular pipeline architecture.
    Currently used for text prompt embeddings, but the blending logic itself is generic.

    Responsibilities:
    - Spatial blending: Combining multiple weighted embeddings into a single embedding
    - Temporal blending: Smooth transitions between embeddings over multiple frames
    - Cache management: Setting conditioning_embeds_updated flag for downstream cache reinitialization
    - Dtype conversion: Ensuring embeddings match pipeline dtype (e.g., bfloat16)
    - State management: Integrating EmbeddingBlender state with pipeline state flow

    Architecture Notes:
    - This block is a thin integration layer around the EmbeddingBlender business logic class
    - EmbeddingBlender remains separate for testability and separation of concerns
    - During transitions, we set conditioning_embeds_updated=True to reset ONLY cross-attention cache,
      preserving KV cache for smooth temporal coherence (unlike init_cache which resets everything)

    Cache Reset Strategy:
    - conditioning_embeds_updated=True → Resets cross-attn cache only (SetupCachesBlock)
    - init_cache=True → Resets ALL caches (KV, cross-attn, VAE, frame counter)
    - We use conditioning_embeds_updated during transitions to maintain temporal context
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
                "prompts",
                type_hint=str | list[str] | list[dict],
                description="Target prompts (received from TextConditioningBlock)",
            ),
            InputParam(
                "current_prompts",
                type_hint=str | list[str] | list[dict] | None,
                description="Current prompts in use (received from TextConditioningBlock)",
            ),
            InputParam(
                "embeds_list",
                type_hint=list[torch.Tensor] | None,
                description="List of pre-encoded embeddings to blend",
            ),
            InputParam(
                "embeds_weights",
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
                "conditioning_embeds",
                type_hint=torch.Tensor,
                description="Existing conditioning embeddings (optional)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "current_prompts",
                type_hint=str | list[str] | list[dict],
                description="Current prompts in use (updated when transition completes)",
            ),
            OutputParam(
                "conditioning_embeds",
                type_hint=torch.Tensor,
                description="Blended embeddings to condition denoising (pipeline state variable)",
            ),
            OutputParam(
                "conditioning_embeds_updated",
                type_hint=bool,
                description="Whether embeddings were updated (requires cross-attention cache re-initialization)",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        if state.get("init_cache", False):
            components.embedding_blender.reset()
            block_state._previous_embeds_signature = None
            block_state.conditioning_embeds = None
            block_state.current_prompts = None

        # Get inputs from state
        embeds_list = block_state.embeds_list
        embeds_weights = block_state.embeds_weights
        spatial_interpolation_method = (
            block_state.spatial_interpolation_method or "linear"
        )
        transition = block_state.transition

        # Detect if conditioning changed by comparing signatures
        current_signature = None
        if embeds_list is not None and embeds_weights is not None:
            # Create signature: (length, weights tuple)
            # This is cheap and avoids storing tensor references
            current_signature = (len(embeds_list), tuple(embeds_weights))

        previous_signature = getattr(block_state, "_previous_embeds_signature", None)
        conditioning_changed = current_signature != previous_signature

        # Update stored signature for next comparison
        block_state._previous_embeds_signature = current_signature

        # Initialize flag
        block_state.conditioning_embeds_updated = False

        # Cancel active transition if conditioning changes mid-transition
        if conditioning_changed and components.embedding_blender.is_transitioning():
            components.embedding_blender.cancel_transition()

        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):
            # Determine transition policy (how to move), independent of prompts
            transition_config = parse_transition_config(transition)

            # Step 1: Spatial blending - compute target embedding when conditioning changes
            target_blend = None
            if embeds_list and embeds_weights and conditioning_changed:
                target_blend = components.embedding_blender.blend(
                    embeddings=embeds_list,
                    weights=embeds_weights,
                    interpolation_method=spatial_interpolation_method,
                    cache_result=False,
                )

            # Step 2: Apply conditioning changes (snap or start transition)
            if conditioning_changed and target_blend is not None:
                has_smooth_transition = transition_config.num_steps > 0

                if has_smooth_transition:
                    source_embedding = getattr(block_state, "conditioning_embeds", None)
                    if source_embedding is None:
                        # No source: snap directly to target
                        block_state.conditioning_embeds = target_blend.to(
                            dtype=components.config.dtype
                        )
                        block_state.conditioning_embeds_updated = True
                    else:
                        # Start temporal transition from source to target
                        components.embedding_blender.start_transition(
                            source_embedding=source_embedding,
                            target_embedding=target_blend,
                            num_steps=transition_config.num_steps,
                            temporal_interpolation_method=transition_config.temporal_interpolation_method,
                        )
                        next_embedding = (
                            components.embedding_blender.get_next_embedding()
                        )
                        if next_embedding is not None:
                            next_embedding = next_embedding.to(
                                dtype=components.config.dtype
                            )
                            block_state.conditioning_embeds = next_embedding
                            block_state.conditioning_embeds_updated = True
                else:
                    # Immediate application (no temporal smoothing)
                    block_state.conditioning_embeds = target_blend.to(
                        dtype=components.config.dtype
                    )
                    block_state.conditioning_embeds_updated = True

            # Step 3: Get next embedding from transition queue (if transitioning and no new change)
            if (
                not conditioning_changed
                and components.embedding_blender.is_transitioning()
            ):
                next_embedding = components.embedding_blender.get_next_embedding()

                if next_embedding is not None:
                    # Cast to pipeline dtype before storing
                    next_embedding = next_embedding.to(dtype=components.config.dtype)
                    block_state.conditioning_embeds = next_embedding
                    block_state.conditioning_embeds_updated = True

        # Signal transition state to frame_processor for lifecycle management
        is_transitioning = components.embedding_blender.is_transitioning()
        state.set("_transition_active", is_transitioning)

        # Update current_prompts when transition completes (contract with TextConditioningBlock)
        # Maintains invariant: current_prompts = active prompts, prompts = target prompts
        if not is_transitioning and block_state.prompts is not None:
            block_state.current_prompts = block_state.prompts

        # Clear transition from PipelineState to prevent reuse
        if transition is not None:
            state.set("transition", None)

        self.set_block_state(state, block_state)
        return components, state
