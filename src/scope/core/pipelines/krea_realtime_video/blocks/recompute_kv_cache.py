import torch
from diffusers.modular_pipelines import BlockState, ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    ConfigSpec,
    InputParam,
    OutputParam,
)
from einops import rearrange

from scope.core.pipelines.wan2_1.utils import initialize_kv_cache


class RecomputeKVCacheBlock(ModularPipelineBlocks):
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
            ConfigSpec("kv_cache_num_frames", 3),
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
            ConfigSpec("local_attn_size", 6),
        ]

    @property
    def description(self) -> str:
        return "Recompute KV Cache block that recomputes KV cache using context frames"

    @staticmethod
    def _get_block_mask_model(model: torch.nn.Module) -> torch.nn.Module:
        """Return the underlying model object that owns `block_mask`.

        When VACE is enabled, `components.generator.model` is wrapped by `CausalVaceWanModel`,
        but the actual `block_mask` lives on the wrapped base model (`causal_wan_model`).
        """
        causal_wan_model = getattr(model, "causal_wan_model", None)
        if causal_wan_model is not None:
            return causal_wan_model

        # PEFT models (runtime_peft) expose the wrapped module at base_model.model.
        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            inner = getattr(base_model, "model", None)
            if inner is not None:
                causal_wan_model = getattr(inner, "causal_wan_model", None)
                if causal_wan_model is not None:
                    return causal_wan_model
                return inner

        return model

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "first_context_frame",
                type_hint=torch.Tensor | None,
                description="First frame of first block",
            ),
            InputParam(
                "context_frame_buffer",
                type_hint=torch.Tensor | None,
                description="Sliding window of latent frames",
            ),
            InputParam(
                "decoded_frame_buffer",
                type_hint=torch.Tensor | None,
                description="Sliding window of decoded frames",
            ),
            InputParam(
                "context_frame_buffer_max_size",
                type_hint=int,
                description="Max size of context frame buffer",
            ),
            InputParam(
                "decoded_frame_buffer_max_size",
                type_hint=int,
                description="Max size of decoded frame buffer",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index of current block",
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
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "start_frame",
                type_hint=torch.Tensor,
                description="Starting frame index that overrides current_start_frame",
            ),
            OutputParam(
                "context_frame_buffer_max_size",
                type_hint=int,
                description="Max size of context frame buffer",
            ),
            OutputParam(
                "decoded_frame_buffer_max_size",
                type_hint=int,
                description="Max size of decoded frame buffer",
            ),
            OutputParam(
                "context_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of latent frames",
            ),
            OutputParam(
                "decoded_frame_buffer",
                type_hint=torch.Tensor,
                description="Sliding window of decoded frames",
            ),
            OutputParam(
                "kv_cache",
                type_hint=list[dict],
                description="Initialized KV cache",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)

        start_frame = min(
            block_state.current_start_frame, components.config.kv_cache_num_frames
        )

        block_state.start_frame = start_frame

        if block_state.current_start_frame == 0:
            context_frame_buffer_max_size = components.config.kv_cache_num_frames - 1
            decoded_frame_buffer_max_size = (
                1 + (components.config.kv_cache_num_frames - 1) * 4
            )

            generator_param = next(components.generator.parameters())

            latent_height = (
                block_state.height // components.config.vae_spatial_downsample_factor
            )
            latent_width = (
                block_state.width // components.config.vae_spatial_downsample_factor
            )
            block_state.context_frame_buffer = torch.zeros(
                [
                    1,
                    context_frame_buffer_max_size,
                    16,
                    latent_height,
                    latent_width,
                ],
                dtype=generator_param.dtype,
                device=generator_param.device,
            )

            block_state.decoded_frame_buffer = torch.zeros(
                [
                    1,
                    decoded_frame_buffer_max_size,
                    3,
                    block_state.height,
                    block_state.width,
                ],
                dtype=generator_param.dtype,
                device=generator_param.device,
            )

            block_state.context_frame_buffer_max_size = context_frame_buffer_max_size
            block_state.decoded_frame_buffer_max_size = decoded_frame_buffer_max_size

            self.set_block_state(state, block_state)
            return components, state

        scale_size = (
            components.config.vae_spatial_downsample_factor
            * components.config.patch_embedding_spatial_downsample_factor
        )
        frame_seq_length = (block_state.height // scale_size) * (
            block_state.width // scale_size
        )

        context_frames = get_context_frames(components, block_state)
        num_context_frames = context_frames.shape[1]

        block_state.kv_cache = initialize_kv_cache(
            generator=components.generator,
            batch_size=1,
            dtype=context_frames.dtype,
            device=context_frames.device,
            local_attn_size=components.config.local_attn_size,
            frame_seq_length=frame_seq_length,
            kv_cache_existing=block_state.kv_cache,
            # The recompute pass immediately overwrites the active cache window and
            # updates indices, so zeroing the full cache is wasted bandwidth.
            zero_cache=False,
        )

        # Prepare blockwise causal mask
        block_mask_model = self._get_block_mask_model(components.generator.model)
        block_mask_model.block_mask = (
            block_mask_model._prepare_blockwise_causal_attn_mask(
                device=context_frames.device,
                num_frames=num_context_frames,
                frame_seqlen=frame_seq_length,
                num_frame_per_block=components.config.num_frame_per_block,
                local_attn_size=-1,
            )
        )

        context_timestep = torch.zeros(
            [1, num_context_frames],
            device=context_frames.device,
            dtype=torch.int64,
        )

        # Cache recomputation: no bias to faithfully store context frames
        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}
        components.generator(
            noisy_image_or_video=context_frames,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=block_state.kv_cache,
            crossattn_cache=block_state.crossattn_cache,
            current_start=start_frame * frame_seq_length,
        )

        block_mask_model.block_mask = None

        self.set_block_state(state, block_state)
        return components, state


def get_context_frames(components, state: BlockState) -> torch.Tensor:
    generator_device = next(components.generator.model.parameters()).device
    if (
        state.current_start_frame - components.config.num_frame_per_block
    ) < components.config.kv_cache_num_frames:
        if components.config.kv_cache_num_frames == 1:
            # The context just contains the first frame
            return state.first_context_frame
        else:
            # The context contains first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
            return torch.cat(
                [
                    state.first_context_frame,
                    state.context_frame_buffer.to(generator_device),
                ],
                dim=1,
            )
    else:
        # The context contains the re-encoded first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
        vae_device = next(components.vae.parameters()).device
        decoded_first_frame = state.decoded_frame_buffer[:, :1].to(vae_device)
        reencoded_latent = components.vae.encode_to_latent(
            rearrange(decoded_first_frame, "B T C H W -> B C T H W"), use_cache=False
        )
        return torch.cat(
            [reencoded_latent, state.context_frame_buffer.to(generator_device)],
            dim=1,
        )
