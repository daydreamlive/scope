import logging
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)

from ...blending import PromptBlender, handle_transition_prepare

logger = logging.getLogger(__name__)


class PromptBlendingBlock(ModularPipelineBlocks):
    """Prompt Blending block that handles spatial and temporal prompt blending.

    This block orchestrates the PromptBlender component within the modular pipeline architecture.

    Responsibilities:
    - Spatial blending: Combining multiple weighted prompts into a single embedding
    - Temporal blending: Smooth transitions between prompt sets over multiple frames
    - Cache management: Setting prompt_embeds_updated flag for downstream cache reinitialization
    - Dtype conversion: Ensuring embeddings match pipeline dtype (e.g., bfloat16)
    - State management: Integrating PromptBlender state with pipeline state flow

    Architecture Notes:
    - This block is a thin integration layer around the PromptBlender business logic class
    - PromptBlender remains separate for testability and separation of concerns
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
            ComponentSpec("prompt_blender", PromptBlender),
            ComponentSpec("text_encoder", torch.nn.Module),
        ]

    @property
    def description(self) -> str:
        return "Prompt Blending block that handles spatial and temporal prompt blending"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompts",
                type_hint=list[dict] | None,
                description="List of prompt dicts with 'text' and 'weight' keys",
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

        prompts = block_state.prompts
        prompt_interpolation_method = (
            block_state.prompt_interpolation_method or "linear"
        )
        transition = block_state.transition

        logger.info(
            f"__call__: Starting with prompts={prompts}, transition={transition is not None}, is_transitioning={components.prompt_blender.is_transitioning()}"
        )

        # Initialize flag
        block_state.prompt_embeds_updated = False

        # Wrap all text encoder calls with autocast to ensure dtype compatibility
        with torch.autocast(
            str(components.config.device), dtype=components.config.dtype
        ):
            # Handle transition requests (temporal blending)
            # Only process transition if we're NOT already transitioning to prevent restart loops
            if (
                transition is not None
                and not components.prompt_blender.is_transitioning()
            ):
                logger.info(f"__call__: Processing transition request: {transition}")
                should_prepare, target_prompts = handle_transition_prepare(
                    transition, components.prompt_blender, components.text_encoder
                )
                logger.info(
                    f"__call__: Transition result: should_prepare={should_prepare}, target_prompts={target_prompts}, is_transitioning={components.prompt_blender.is_transitioning()}"
                )

                # If transition is immediate (num_steps=0), update prompts directly
                if should_prepare and target_prompts:
                    logger.info("__call__: Applying immediate transition (num_steps=0)")
                    prompts = target_prompts
                    block_state.prompt_embeds_updated = True
            elif (
                transition is not None and components.prompt_blender.is_transitioning()
            ):
                logger.info(
                    "__call__: Ignoring transition request - already transitioning"
                )

            # Check if prompts or interpolation method changed
            should_update = components.prompt_blender.should_update(
                prompts, prompt_interpolation_method
            )
            logger.info(
                f"__call__: should_update={should_update}, is_transitioning={components.prompt_blender.is_transitioning()}"
            )

            if should_update:
                logger.info(
                    "__call__: Prompts or interpolation method changed, updating blend"
                )

                # Blend the prompts (spatial blending)
                blended_embeds = components.prompt_blender.blend(
                    prompts, prompt_interpolation_method, components.text_encoder
                )

                logger.info(f"__call__: blend() returned: {blended_embeds is not None}")

                # Only update if blend succeeded (returns None during transitions)
                if blended_embeds is not None:
                    block_state.prompt_embeds = blended_embeds.to(
                        dtype=components.config.dtype
                    )
                    block_state.prompt_embeds_updated = True

            # Get next embedding (handles both normal blending and transitions)
            # NOTE: During transitions, this returns interpolated embeddings
            # The PromptBlender's internal callback resets cross-attn cache (via init_cache=True)
            # We need to override this by setting prompt_embeds_updated=True which ONLY resets cross-attn cache
            logger.info(
                f"__call__: Calling get_next_embedding(), is_transitioning={components.prompt_blender.is_transitioning()}"
            )
            is_transitioning_before = components.prompt_blender.is_transitioning()
            next_embedding = components.prompt_blender.get_next_embedding(
                components.text_encoder
            )
            is_transitioning_after = components.prompt_blender.is_transitioning()

            logger.info(
                f"__call__: get_next_embedding() returned embedding: {next_embedding is not None}"
            )

            if next_embedding is not None:
                # Cast to pipeline dtype before storing
                next_embedding = next_embedding.to(dtype=components.config.dtype)

                # During transitions, set prompt_embeds_updated=True to reset ONLY cross-attn cache
                # This prevents the full cache reset (init_cache=True) from the PromptBlender callback
                if is_transitioning_before or is_transitioning_after:
                    block_state.prompt_embeds = next_embedding
                    block_state.prompt_embeds_updated = True
                    logger.info(
                        "__call__: Updated prompt_embeds from transition step (cross-attn cache reset only)"
                    )
                else:
                    block_state.prompt_embeds = next_embedding
                    logger.info(
                        "__call__: Updated prompt_embeds from get_next_embedding() (no cache reset)"
                    )

        logger.info(
            f"__call__: Finished, prompt_embeds_updated={block_state.prompt_embeds_updated}, is_transitioning={components.prompt_blender.is_transitioning()}"
        )

        self.set_block_state(state, block_state)
        return components, state
