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


class CleanKVCacheBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("record_interval", 3),
        ]

    @property
    def description(self) -> str:
        return "Clean KV Cache block runs the generator with timestep zero and denoised latents to clean the KV cache"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
            ),
            InputParam(
                "kv_cache_manager",
                required=True,
                description="KVCacheManager (ring buffer)",
            ),
            InputParam(
                "height", required=True, type_hint=int, description="Height of video"
            ),
            InputParam(
                "width", required=True, type_hint=int, description="Width of video"
            ),
            InputParam(
                "conditioning_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditioning embeddings to condition denoising",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return []

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        generator_param = next(components.generator.parameters())

        _, num_frames, _, _, _ = block_state.latents.shape

        # Compute cache metadata for the clean pass (timestep=0)
        cache_mgr = block_state.kv_cache_manager
        num_new_tokens = num_frames * frame_seq_length
        h = block_state.height // scale_size
        w = block_state.width // scale_size
        freqs = components.generator.model.freqs.to(generator_param.device)
        write_indices, attn_mask, rope_freqs = cache_mgr.prepare_step(
            num_new_tokens=num_new_tokens,
            start_frame=block_state.current_start_frame,
            freqs=freqs,
            f=num_frames,
            h=h,
            w=w,
        )

        context_timestep = (
            torch.ones(
                [1, num_frames],
                device=generator_param.device,
                dtype=torch.int64,
            )
            * 0
        )

        # Run generator at timestep=0 to write clean K/V into cache
        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}
        components.generator(
            noisy_image_or_video=block_state.latents,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            write_indices=write_indices,
            attn_mask=attn_mask,
            rope_freqs=rope_freqs,
        )
        # Note: we do NOT advance the ring pointer here because we're
        # overwriting the same positions that were just written during denoise.

        self.set_block_state(state, block_state)
        return components, state
