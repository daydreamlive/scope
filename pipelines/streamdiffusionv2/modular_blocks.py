from typing import Any

import torch
from diffusers.modular_pipelines import PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.modular_pipeline_utils import (
    InputParam,
    InsertableDict,
)
from diffusers.utils import logging as diffusers_logging

from ..wan2_1.blocks import (
    CleanKVCacheBlock,
    DecodeBlock,
    EmbeddingBlendingBlock,
    NoiseScaleControllerBlock,
    PrepareNextBlock,
    PrepareVideoLatentsBlock,
    PreprocessVideoBlock,
    SetTimestepsBlock,
    SetupCachesBlock,
    TextConditioningBlock,
)
from ..wan2_1.blocks import (
    DenoiseBlock as BaseDenoiseBlock,
)

logger = diffusers_logging.get_logger(__name__)


class StreamDiffusionV2DenoiseBlock(BaseDenoiseBlock):
    @property
    def inputs(self) -> list[InputParam]:
        # Get base inputs
        inputs = super().inputs
        # Add I2V specific inputs
        inputs.extend(
            [
                InputParam(
                    "i2v_conditioning_latent",
                    type_hint=torch.Tensor | None,
                    description="Latent representation of the conditioning image for I2V",
                ),
                InputParam(
                    "clip_conditioning_scale",
                    type_hint=float,
                    default=1.0,
                    description="Scale factor for CLIP conditioning features (1.0 = full strength)",
                ),
            ]
        )
        return inputs

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

        # NOTE: Commented out to allow manual denoising_step_list control
        # The noise_scale parameter is still used in PrepareVideoLatentsBlock for initial noise mixing
        # if block_state.noise_scale is not None:
        #     # Higher noise scale -> more denoising steps, more intense changes to input
        #     # Lower noise scale -> less denoising steps, less intense changes to input
        #     denoising_step_list[0] = int(1000 * block_state.noise_scale) - 100

        # Check model type to determine if 'y' is required
        model_type = getattr(components.generator.model, "model_type", "t2v")

        # CFG for CLIP conditioning: only apply when scale != 1.0 and CLIP features exist
        clip_features = block_state.clip_features

        # Disable CLIP features for T2V model as it uses random projection weights
        if model_type == "t2v" and clip_features is not None:
            logger.warning("Disabling CLIP features for T2V model (unsupported)")
            clip_features = None

        use_cfg = clip_features is not None and block_state.clip_conditioning_scale != 1.0

        if clip_features is not None:
            logger.info(f"CLIP features present, scale={block_state.clip_conditioning_scale}, use_cfg={use_cfg}")

        # I2V Conditioning Latent (Channel Concatenation)
        # "y" in the model forward pass
        i2v_latents = getattr(block_state, "i2v_conditioning_latent", None)

        y = None
        if model_type == "i2v":
            # TODO: This was a test of i2v - it's been clarified that training is required - not implemented at the moment
            # I2V model expects 'y' (20 channels: 16 latent + 4 mask)

            if i2v_latents is None:
                # T2V mode with I2V model: Provide zero-filled conditioning
                # Shape: [B, 20, T, H, W]
                y = torch.zeros(
                    batch_size,
                    20,
                    num_frames,
                    block_state.height // scale_size,
                    block_state.width // scale_size,
                    device=noise.device,
                    dtype=noise.dtype,
                )
            else:
                # I2V mode: Use provided latents
                # i2v_latents: [1, 20, 1, H, W]

                # 1. Handle Batch Dimension
                if i2v_latents.shape[0] != batch_size:
                    i2v_latents = i2v_latents.repeat(batch_size, 1, 1, 1, 1)

                # 2. Handle Temporal Dimension (Expand to current chunk size)
                # We repeat the single frame latent for the entire chunk
                if i2v_latents.shape[2] != num_frames:
                    i2v_latents = i2v_latents.repeat(1, 1, num_frames, 1, 1)

                # 3. Handle Spatial Dimensions (Resize if needed)
                target_h = block_state.height // scale_size
                target_w = block_state.width // scale_size
                if i2v_latents.shape[3] != target_h or i2v_latents.shape[4] != target_w:
                    logger.warning(
                        f"Resizing i2v_latents from {i2v_latents.shape[3:]} to {(target_h, target_w)}"
                    )
                    # Reshape for interpolation: [B, C, T, H, W] -> [B*T, C, H, W]
                    b, c, t, h, w = i2v_latents.shape
                    i2v_latents = i2v_latents.permute(0, 2, 1, 3, 4).reshape(
                        b * t, c, h, w
                    )
                    i2v_latents = torch.nn.functional.interpolate(
                        i2v_latents, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                    i2v_latents = i2v_latents.reshape(
                        b, t, c, target_h, target_w
                    ).permute(0, 2, 1, 3, 4)

                # 4. Apply Scaling (Control influence)
                # Use clip_conditioning_scale as a global "Image Influence" slider
                if block_state.clip_conditioning_scale != 1.0:
                    i2v_latents = i2v_latents * block_state.clip_conditioning_scale

                y = i2v_latents

        # Denoising loop
        for index, current_timestep in enumerate(denoising_step_list):
            timestep = (
                torch.ones(
                    [batch_size, num_frames],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(denoising_step_list) - 1:
                '''
                if use_cfg:
                    # CFG: Run model twice - with and without CLIP features
                    _, cond_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=clip_features,
                        y=y,
                    )
                    _, uncond_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=None,
                        y=y,
                    )
                    # CFG formula: uncond + scale * (cond - uncond)
                    scale = block_state.clip_conditioning_scale
                    denoised_pred = uncond_pred + scale * (cond_pred - uncond_pred)
                else:
                    _, denoised_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=clip_features,
                        y=y,
                    )
                '''
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    clip_features=clip_features,
                    y=y,
                )
                next_timestep = denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                # denoised_pred: [B, T, C, H, W]
                # We need to flatten B and T to match timestep: [B*T]
                flattened_pred = denoised_pred.flatten(0, 1)

                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=block_state.generator,
                )

                # Add noise
                # Output noise: [B*T, C, H, W]
                noise = components.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [batch_size * num_frames],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                )

                # Restore shape: [B*T, C, H, W] -> [B, T, C, H, W]
                noise = noise.unflatten(0, (batch_size, num_frames))
            else:
                '''
                if use_cfg:
                    # CFG: Run model twice - with and without CLIP features
                    _, cond_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=clip_features,
                        y=y,
                    )
                    _, uncond_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=None,
                        y=y,
                    )
                    scale = block_state.clip_conditioning_scale
                    denoised_pred = uncond_pred + scale * (cond_pred - uncond_pred)
                else:
                    _, denoised_pred = components.generator(
                        noisy_image_or_video=noise,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=block_state.kv_cache,
                        crossattn_cache=block_state.crossattn_cache,
                        current_start=start_frame * frame_seq_length,
                        current_end=end_frame * frame_seq_length,
                        kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                        clip_features=clip_features,
                        y=y,
                    )
                '''
                _, denoised_pred = components.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=block_state.kv_cache,
                    crossattn_cache=block_state.crossattn_cache,
                    current_start=start_frame * frame_seq_length,
                    current_end=end_frame * frame_seq_length,
                    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
                    clip_features=clip_features,
                    y=y,
                )

        block_state.latents = denoised_pred

        self.set_block_state(state, block_state)
        return components, state


# Main pipeline blocks for V2V workflow
ALL_BLOCKS = InsertableDict(
    [
        ("text_conditioning", TextConditioningBlock),
        ("embedding_blending", EmbeddingBlendingBlock),
        ("set_timesteps", SetTimestepsBlock),
        ("preprocess_video", PreprocessVideoBlock),
        ("noise_scale_controller", NoiseScaleControllerBlock),
        ("setup_caches", SetupCachesBlock),
        ("prepare_video_latents", PrepareVideoLatentsBlock),
        ("denoise", StreamDiffusionV2DenoiseBlock),  # Use the new block
        ("clean_kv_cache", CleanKVCacheBlock),
        ("decode", DecodeBlock),
        ("prepare_next", PrepareNextBlock),
    ]
)


class StreamDiffusionV2Blocks(SequentialPipelineBlocks):
    block_classes = list(ALL_BLOCKS.values())
    block_names = list(ALL_BLOCKS.keys())
