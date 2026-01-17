# Enhanced DenoiseBlock with FreSca and TSR support
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

from ..enhancements import (
    apply_step_adaptive_scaling,
    fourier_filter,
    normalized_fourier_filter,
    temporal_score_rescaling,
)


class EnhancedDenoiseBlock(ModularPipelineBlocks):
    """
    Enhanced denoising block with FreSca and TSR support.

    Enhancements:
    - FreSca: Frequency-selective scaling for detail enhancement
    - TSR: Temporal score rescaling for improved temporal coherence
    """

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("generator", torch.nn.Module),
            ComponentSpec("scheduler", torch.nn.Module),
        ]

    @property
    def expected_configs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec("patch_embedding_spatial_downsample_factor", 2),
            ConfigSpec("vae_spatial_downsample_factor", 8),
        ]

    @property
    def description(self) -> str:
        return "Enhanced denoise block with FreSca and TSR enhancements"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "height",
                type_hint=int,
                description="Height of the video",
            ),
            InputParam(
                "width",
                type_hint=int,
                description="Width of the video",
            ),
            InputParam(
                "kv_cache_attention_bias",
                default=1.0,
                type_hint=float,
                description="Controls reliance on past frames in cache",
            ),
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents to denoise",
            ),
            InputParam(
                "current_denoising_step_list",
                required=True,
                type_hint=torch.Tensor,
                description="Current list of denoising steps",
            ),
            InputParam(
                "conditioning_embeds",
                required=True,
                type_hint=torch.Tensor,
                description="Conditioning embeddings",
            ),
            InputParam(
                "current_start_frame",
                required=True,
                type_hint=int,
                description="Current starting frame index",
            ),
            InputParam(
                "start_frame",
                type_hint=int | None,
                description="Starting frame index override",
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
                "kv_bank",
                type_hint=list[dict],
                description="Initialized KV memory bank",
            ),
            InputParam(
                "generator",
                required=True,
                description="Random number generator",
            ),
            InputParam(
                "noise_scale",
                type_hint=float | None,
                description="Amount of noise added to video",
            ),
            InputParam(
                "vace_context",
                default=None,
                type_hint=torch.Tensor | None,
                description="VACE context for visual conditioning",
            ),
            InputParam(
                "vace_context_scale",
                default=1.0,
                type_hint=float,
                description="Scaling factor for VACE hint injection",
            ),
            # Enhancement parameters
            InputParam(
                "enable_fresca",
                default=False,
                type_hint=bool,
                description="Enable FreSca frequency-selective enhancement",
            ),
            InputParam(
                "fresca_scale_low",
                default=1.0,
                type_hint=float,
                description="FreSca low-frequency scaling (structure preservation)",
            ),
            InputParam(
                "fresca_scale_high",
                default=1.15,
                type_hint=float,
                description="FreSca high-frequency scaling (detail enhancement)",
            ),
            InputParam(
                "fresca_freq_cutoff",
                default=20,
                type_hint=int,
                description="FreSca frequency cutoff radius",
            ),
            InputParam(
                "fresca_adaptive",
                default=False,
                type_hint=bool,
                description="Enable step-adaptive FreSca scaling",
            ),
            InputParam(
                "fresca_tau",
                default=None,
                type_hint=float | None,
                description="Normalized FreSca tau (max norm ratio). When set, enables "
                "self-limiting enhancement that prevents accumulation over time. "
                "Typical value: 1.2 means enhanced output is at most 1.2x original norm.",
            ),
            InputParam(
                "enable_tsr",
                default=False,
                type_hint=bool,
                description="Enable Temporal Score Rescaling",
            ),
            InputParam(
                "tsr_k",
                default=0.95,
                type_hint=float,
                description="TSR sampling temperature (0.9-1.0 typical)",
            ),
            InputParam(
                "tsr_sigma",
                default=0.1,
                type_hint=float,
                description="TSR SNR influence factor",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Denoised latents",
            ),
        ]

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

        noise = block_state.latents
        batch_size = noise.shape[0]
        num_frames = noise.shape[1]
        denoising_step_list = block_state.current_denoising_step_list.clone()

        conditional_dict = {"prompt_embeds": block_state.conditioning_embeds}

        start_frame = block_state.current_start_frame
        if block_state.start_frame is not None:
            start_frame = block_state.start_frame

        end_frame = start_frame + num_frames

        if block_state.noise_scale is not None:
            denoising_step_list[0] = int(1000 * block_state.noise_scale) - 100

        # Get enhancement parameters with explicit None handling
        # (block_state may return None for unset attributes instead of raising AttributeError)
        enable_fresca = getattr(block_state, "enable_fresca", False) or False
        fresca_scale_low = getattr(block_state, "fresca_scale_low", None)
        fresca_scale_high = getattr(block_state, "fresca_scale_high", None)
        fresca_freq_cutoff = getattr(block_state, "fresca_freq_cutoff", None)
        fresca_adaptive = getattr(block_state, "fresca_adaptive", False) or False
        fresca_tau = getattr(block_state, "fresca_tau", None)

        # Apply defaults for None values
        if fresca_scale_low is None:
            fresca_scale_low = 1.0
        if fresca_scale_high is None:
            fresca_scale_high = 1.15
        if fresca_freq_cutoff is None:
            fresca_freq_cutoff = 20

        enable_tsr = getattr(block_state, "enable_tsr", False) or False
        tsr_k = getattr(block_state, "tsr_k", None)
        tsr_sigma = getattr(block_state, "tsr_sigma", None)

        # Apply defaults for None values
        if tsr_k is None:
            tsr_k = 0.95
        if tsr_sigma is None:
            tsr_sigma = 0.1

        total_steps = len(denoising_step_list)

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            if index == 0:
                q_bank = True
            else:
                q_bank = False

            timestep = (
                torch.ones(
                    [batch_size, num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(denoising_step_list) - 1:
                flow_pred, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    kv_bank=block_state.kv_bank,
                    update_bank=False,
                    q_bank=q_bank,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    vace_context=block_state.vace_context,
                    vace_context_scale=block_state.vace_context_scale,
                )

                # Apply TSR to flow prediction if enabled
                if enable_tsr:
                    flow_pred = temporal_score_rescaling(
                        model_output=flow_pred,
                        sample=noise,
                        timestep=current_timestep,
                        k=tsr_k,
                        tsr_sigma=tsr_sigma,
                    )
                    # Recompute denoised_pred from modified flow_pred
                    denoised_pred = components.generator._convert_flow_pred_to_x0(
                        flow_pred=flow_pred.flatten(0, 1),
                        xt=noise.flatten(0, 1),
                        timestep=timestep.flatten(0, 1),
                    ).unflatten(0, flow_pred.shape[:2])

                # Apply FreSca to denoised prediction if enabled
                if enable_fresca:
                    if fresca_adaptive:
                        denoised_pred = apply_step_adaptive_scaling(
                            denoised_pred,
                            step_index=index,
                            total_steps=total_steps,
                            scale_low=fresca_scale_low,
                            scale_high_start=1.0,
                            scale_high_end=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                        )
                    elif fresca_tau is not None:
                        # Normalized FreSca: self-limiting to prevent accumulation
                        denoised_pred = normalized_fourier_filter(
                            denoised_pred,
                            scale_low=fresca_scale_low,
                            scale_high=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                            tau=fresca_tau,
                        )
                    else:
                        denoised_pred = fourier_filter(
                            denoised_pred,
                            scale_low=fresca_scale_low,
                            scale_high=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                        )

                next_timestep = denoising_step_list[index + 1]
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=block_state.generator,
                )
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # Final step
                flow_pred, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    kv_bank=block_state.kv_bank,
                    update_bank=False,
                    q_bank=q_bank,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    vace_context=block_state.vace_context,
                    vace_context_scale=block_state.vace_context_scale,
                )

                # Apply TSR on final step too
                if enable_tsr:
                    flow_pred = temporal_score_rescaling(
                        model_output=flow_pred,
                        sample=noise,
                        timestep=current_timestep,
                        k=tsr_k,
                        tsr_sigma=tsr_sigma,
                    )
                    denoised_pred = components.generator._convert_flow_pred_to_x0(
                        flow_pred=flow_pred.flatten(0, 1),
                        xt=noise.flatten(0, 1),
                        timestep=timestep.flatten(0, 1),
                    ).unflatten(0, flow_pred.shape[:2])

                # Apply FreSca on final output
                if enable_fresca:
                    if fresca_adaptive:
                        denoised_pred = apply_step_adaptive_scaling(
                            denoised_pred,
                            step_index=index,
                            total_steps=total_steps,
                            scale_low=fresca_scale_low,
                            scale_high_start=1.0,
                            scale_high_end=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                        )
                    elif fresca_tau is not None:
                        # Normalized FreSca: self-limiting to prevent accumulation
                        denoised_pred = normalized_fourier_filter(
                            denoised_pred,
                            scale_low=fresca_scale_low,
                            scale_high=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                            tau=fresca_tau,
                        )
                    else:
                        denoised_pred = fourier_filter(
                            denoised_pred,
                            scale_low=fresca_scale_low,
                            scale_high=fresca_scale_high,
                            freq_cutoff=fresca_freq_cutoff,
                        )

        block_state.latents = denoised_pred

        self.set_block_state(state, block_state)
        return components, state
