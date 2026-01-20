import logging
from typing import Any

import torch
from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)

from ..utils import initialize_crossattn_cache, initialize_kv_cache

logger = logging.getLogger(__name__)


class SetupCachesBlock(ModularPipelineBlocks):
    model_name = "Wan2.1"

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("vae", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("local_attn_size", 6),
        ]

    @property
    def description(self) -> str:
        return "Setup Caches block makes sure all caches are setup"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "kv_cache",
                type_hint=list[dict] | None,
                default=None,
                description="Existing KV cache",
            ),
            InputParam(
                "crossattn_cache",
                type_hint=list[dict] | None,
                default=None,
                description="Existing cross-attention cache",
            ),
            InputParam(
                "init_cache",
                type_hint=bool,
                default=False,
                description="Whether to (re)initialize caches",
            ),
            InputParam(
                "conditioning_embeds_updated",
                type_hint=bool,
                default=False,
                description="Whether conditioning embeddings were updated (requires cross-attention cache re-initialization)",
            ),
            InputParam(
                "height",
                required=True,
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                required=True,
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index for current block",
            ),
            InputParam(
                "manage_cache",
                type_hint=bool,
                default=True,
                description="Whether cache management is enabled",
            ),
            InputParam(
                "video",
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                default=None,
                description="Input video (if present, indicates video input is enabled)",
            ),
            InputParam(
                "vace_input_frames",
                type_hint=list[torch.Tensor] | torch.Tensor | None,
                default=None,
                description="Input frames for VACE conditioning (if present, indicates video input is enabled)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "kv_cache",
                type_hint=list[dict],
                description="Initialized KV cache",
            ),
            OutputParam(
                "crossattn_cache",
                type_hint=list[dict],
                description="Initialized cross-attention cache",
            ),
            OutputParam(
                "current_start_frame",
                type_hint=int,
                description="Current starting frame index for current block",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        init_cache = block_state.init_cache
        is_transitioning = state.get("_transition_active", False)
        was_transitioning = state.get("_transition_active_prev", False)

        if init_cache:
            logger.info(
                "HARD CUT: SetupCachesBlock received init_cache=True, will reset KV cache"
            )

        max_current_start = (
            components.config.max_rope_freq_table_seq_len
            - components.config.num_frame_per_block
        )
        if block_state.current_start_frame >= max_current_start:
            init_cache = True

        # Clear KV cache when conditioning changes outside of a transition if manage_cache is enabled and video input is present.
        # Transitions include: embedding blending (_transition_active) and soft transitions (KV bias lowering).
        transitioning_context = is_transitioning or was_transitioning
        if (
            block_state.conditioning_embeds_updated
            and not transitioning_context
            and block_state.manage_cache
            and (
                block_state.video is not None
                or block_state.vace_input_frames is not None
            )
        ):
            init_cache = True

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        generator_param = next(components.generator.model.parameters())
        generator_dtype = generator_param.dtype
        generator_device = generator_param.device

        if init_cache or block_state.kv_cache is None:
            for block in components.generator.model.blocks:
                block.self_attn.local_attn_size = -1
                block.self_attn.num_frame_per_block = (
                    components.config.num_frame_per_block
                )

            components.generator.model.local_attn_size = (
                components.config.local_attn_size
            )

            set_all_modules_frame_seq_length(components.generator, frame_seq_length)
            set_all_modules_max_attention_size(
                components.generator,
                components.config.local_attn_size * frame_seq_length,
            )

            block_state.kv_cache = initialize_kv_cache(
                generator=components.generator,
                batch_size=1,
                dtype=generator_dtype,
                device=generator_device,
                local_attn_size=components.config.local_attn_size,
                frame_seq_length=frame_seq_length,
                kv_cache_existing=block_state.kv_cache,
            )

        # If the conditioning embeds change we need to reinitialize the crossattn cache
        # During transitions, this updates cross-attn cache without full KV cache reset
        if (
            init_cache
            or block_state.crossattn_cache is None
            or block_state.conditioning_embeds_updated
        ):
            block_state.crossattn_cache = initialize_crossattn_cache(
                generator=components.generator,
                batch_size=1,
                dtype=generator_param.dtype,
                device=generator_param.device,
                crossattn_cache_existing=block_state.crossattn_cache,
            )

        if init_cache:
            logger.info(
                "HARD CUT: Executing cache reset - current_start_frame=0, clearing VAE cache"
            )
            block_state.current_start_frame = 0

            components.vae.clear_cache()

        state.set("_transition_active_prev", is_transitioning)
        self.set_block_state(state, block_state)
        return components, state


def set_all_modules_max_attention_size(generator, max_attention_size: int):
    """
    Set max_attention_size on all submodules that define it.
    """
    updated_modules = []
    # Update root model if applicable
    if hasattr(generator.model, "max_attention_size"):
        generator.model.max_attention_size = max_attention_size
        updated_modules.append("<root_model>")

    # Update all child modules
    for name, module in generator.model.named_modules():
        if hasattr(module, "max_attention_size"):
            module.max_attention_size = max_attention_size
            updated_modules.append(name if name else module.__class__.__name__)


def set_all_modules_frame_seq_length(generator, frame_seq_length: int):
    """
    Set frame_seq_length on all submodules that define it.
    """
    if hasattr(generator, "seq_len") and hasattr(generator.model, "local_attn_size"):
        local_attn_size = generator.model.local_attn_size
        if local_attn_size > 21:
            generator.seq_len = frame_seq_length * local_attn_size
        else:
            generator.seq_len = 32760

    # Update root model if applicable
    if hasattr(generator.model, "frame_seq_length"):
        generator.model.frame_seq_length = frame_seq_length

    # Update all child modules (especially CausalWanSelfAttention instances)
    for _, module in generator.model.named_modules():
        if hasattr(module, "frame_seq_length"):
            module.frame_seq_length = frame_seq_length
