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
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
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
                "kv_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized KV cache",
            ),
            InputParam(
                "kv_bank",
                type_hint=list[dict],
                description="Initialized KV memory bank",
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
                description="Whether conditioning_embeds were updated (requires frame recaching)",
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
            # Initialize recache buffer
            latent_height = (
                block_state.height // components.config.vae_spatial_downsample_factor
            )
            latent_width = (
                block_state.width // components.config.vae_spatial_downsample_factor
            )
            block_state.recache_buffer = torch.zeros(
                [
                    1,
                    components.config.local_attn_size,
                    16,
                    latent_height,
                    latent_width,
                ],
                dtype=generator_param.dtype,
                device=generator_param.device,
            )

            self.set_block_state(state, block_state)
            return components, state

        # Only recache frames if conditioning_embeds were updated
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
        num_frame_per_block = components.config.num_frame_per_block
        sink_tokens = components.generator.sink_tokens

        # Preserve the sink frames (old scene continuity) but zero non-sink portion.
        # This way chunked recache has the sink as context for smooth transitions.
        for cache in block_state.kv_cache:
            cache["k"][:, sink_tokens:].zero_()
            cache["v"][:, sink_tokens:].zero_()

        # Set fill_level to sink_tokens so only the sink is valid context initially.
        # Each chunk will grow fill_level as it processes.
        components.generator.fill_level = sink_tokens

        # Get the number of frames to recache (min of what we've generated and buffer size)
        num_recache_frames = min(
            block_state.current_start_frame, components.config.local_attn_size
        )
        recache_start = block_state.current_start_frame - num_recache_frames
        recache_frames = (
            block_state.recache_buffer[:, -num_recache_frames:]
            .contiguous()
            .to(generator_param.device)
        )

        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}

        # Process recache frames in chunks of num_frame_per_block.
        # Each chunk sees the sink (old scene continuity) + previously recached chunks
        # as context, maintaining smooth transitions between prompts.
        for chunk_idx in range(0, num_recache_frames, num_frame_per_block):
            chunk_end = min(chunk_idx + num_frame_per_block, num_recache_frames)
            chunk_frames = recache_frames[:, chunk_idx:chunk_end]

            chunk_num_frames = chunk_end - chunk_idx
            chunk_start_frame = recache_start + chunk_idx

            context_timestep = (
                torch.ones(
                    [1, chunk_num_frames],
                    device=chunk_frames.device,
                    dtype=torch.int64,
                )
                * 0
            )

            components.generator(
                noisy_image_or_video=chunk_frames,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=block_state.kv_cache,
                kv_bank=block_state.kv_bank,
                update_bank=False,
                q_bank=True,
                update_cache=True,
                current_start=chunk_start_frame * frame_seq_length,
            )

        self.set_block_state(state, block_state)
        return components, state
