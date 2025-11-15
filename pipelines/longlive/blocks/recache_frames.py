import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)

from ...wan2_1.utils import initialize_crossattn_cache, initialize_kv_cache


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
                "crossattn_cache",
                required=True,
                type_hint=list[dict],
                description="Initialized cross-attention cache",
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
                "crossattn_cache",
                type_hint=list[dict],
                description="Initialized cross-attention cache",
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

        block_state.kv_cache = initialize_kv_cache(
            generator=components.generator,
            batch_size=1,
            dtype=generator_param.dtype,
            device=generator_param.device,
            local_attn_size=components.config.local_attn_size,
            frame_seq_length=frame_seq_length,
            kv_cache_existing=block_state.kv_cache,
            reset_indices=False,  # Do not reset indices
        )

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

        # Prepare blockwise causal mask
        components.generator.model.block_mask = (
            components.generator.model._prepare_blockwise_causal_attn_mask(
                device=recache_frames.device,
                num_frames=num_recache_frames,
                frame_seqlen=frame_seq_length,
                num_frame_per_block=components.config.num_frame_per_block,
                local_attn_size=components.config.local_attn_size,
            )
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
            kv_cache=block_state.kv_cache,
            crossattn_cache=block_state.crossattn_cache,
            current_start=recache_start * frame_seq_length,
        )

        block_state.crossattn_cache = initialize_crossattn_cache(
            generator=components.generator,
            batch_size=1,
            dtype=generator_param.dtype,
            device=generator_param.device,
            crossattn_cache_existing=block_state.crossattn_cache,
        )

        self.set_block_state(state, block_state)
        return components, state
