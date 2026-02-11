import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)


class RecacheFramesBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("num_frame_per_block", 3),
            ConfigSpec("local_attn_size", 12),
            ConfigSpec("global_sink", True),
        ]

    @property
    def description(self) -> str:
        return "Recache Frames block that recaches frames in the KV cache"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
            ),
            InputParam(
                "recache_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of recache frames",
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
            InputParam(
                "conditioning_embeds_updated",
                required=True,
                type_hint=bool,
                description="Whether conditioning_embeds were updated",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "kv_cache_manager",
                description="KVCacheManager (ring buffer)",
            ),
            OutputParam(
                "recache_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of recache frames",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        generator_param = next(components.generator.model.parameters())

        if block_state.current_start_frame == 0:
            latent_height = (
                block_state.height // components.config.vae_spatial_downsample_factor
            )
            latent_width = (
                block_state.width // components.config.vae_spatial_downsample_factor
            )
            block_state.recache_buffer = torch.zeros(
                [1, components.config.local_attn_size, 16, latent_height, latent_width],
                dtype=generator_param.dtype,
                device=generator_param.device,
            )
            self.set_block_state(state, block_state)
            return components, state

        if not block_state.conditioning_embeds_updated:
            self.set_block_state(state, block_state)
            return components, state

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        global_sink = components.config.global_sink
        cache_mgr = block_state.kv_cache_manager
        cache_mgr.reset_for_recache(keep_sink=global_sink)

        num_recache_frames = min(
            block_state.current_start_frame, components.config.local_attn_size
        )
        recache_start = block_state.current_start_frame - num_recache_frames
        recache_frames = (
            block_state.recache_buffer[:, -num_recache_frames:]
            .contiguous()
            .to(generator_param.device)
        )

        num_recache_tokens = num_recache_frames * frame_seq_length
        h = block_state.height // scale_size
        w = block_state.width // scale_size
        freqs = components.generator.model.freqs.to(generator_param.device)

        write_indices, attn_mask, rope_freqs = cache_mgr.prepare_step(
            num_new_tokens=num_recache_tokens,
            start_frame=recache_start,
            freqs=freqs,
            f=num_recache_frames,
            h=h,
            w=w,
        )

        context_timestep = (
            torch.ones(
                [1, num_recache_frames],
                device=recache_frames.device,
                dtype=torch.int64,
            )
            * 0
        )

        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}
        components.generator(
            noisy_image_or_video=recache_frames,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            write_indices=write_indices,
            attn_mask=attn_mask,
            rope_freqs=rope_freqs,
        )

        cache_mgr.advance(num_recache_tokens)

        self.set_block_state(state, block_state)
        return components, state
