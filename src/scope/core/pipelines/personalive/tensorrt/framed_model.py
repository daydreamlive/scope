"""Framed model for ONNX export.

This module provides the UNetWork class that bundles multiple models
into a single module for ONNX export and TensorRT conversion.

The bundled models include:
- PoseGuider: Extracts pose features from keypoint images
- MotEncoder: Encodes facial motion features
- UNet3DConditionModel: Main denoising UNet
- VAE decoder: Decodes latents to images
- DDIMScheduler: Denoising scheduler step
"""

import torch
import torch.nn as nn
from einops import rearrange


class UNetWork(nn.Module):
    """Bundled model for TensorRT export.

    This class wraps the pose_guider, motion_encoder, denoising_unet,
    vae decoder, and scheduler into a single module that can be exported
    to ONNX for TensorRT conversion.

    The forward pass handles a complete denoising step including:
    1. Encoding new pose features
    2. Encoding new motion features
    3. Running the UNet denoiser
    4. Scheduler step
    5. VAE decoding for the output frames
    """

    def __init__(
        self,
        pose_guider: nn.Module,
        motion_encoder: nn.Module,
        denoising_unet: nn.Module,
        vae: nn.Module,
        scheduler,
        timesteps: torch.Tensor,
    ):
        """Initialize the bundled model.

        Args:
            pose_guider: PoseGuider model.
            motion_encoder: MotEncoder model.
            denoising_unet: UNet3DConditionModel for denoising.
            vae: AutoencoderKL for decoding.
            scheduler: DDIMScheduler instance.
            timesteps: Fixed timesteps tensor for denoising.
        """
        super().__init__()
        self.pose_guider = pose_guider
        self.motion_encoder = motion_encoder
        self.unet = denoising_unet
        self.vae = vae
        self.scheduler = scheduler
        self.timesteps = timesteps

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE.

        Args:
            latents: Latent tensor of shape (B, C, H, W).

        Returns:
            Image tensor of shape (B, H, W, C) in [0, 1] range.
        """
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        images = rearrange(images, "b c h w -> b h w c")
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def forward(
        self,
        sample: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        motion_hidden_states: torch.Tensor,
        motion: torch.Tensor,
        pose_cond_fea: torch.Tensor,
        pose: torch.Tensor,
        new_noise: torch.Tensor,
        d00: torch.Tensor,
        d01: torch.Tensor,
        d10: torch.Tensor,
        d11: torch.Tensor,
        d20: torch.Tensor,
        d21: torch.Tensor,
        m: torch.Tensor,
        u10: torch.Tensor,
        u11: torch.Tensor,
        u12: torch.Tensor,
        u20: torch.Tensor,
        u21: torch.Tensor,
        u22: torch.Tensor,
        u30: torch.Tensor,
        u31: torch.Tensor,
        u32: torch.Tensor,
    ):
        """Forward pass for the bundled model.

        Args:
            sample: Noisy latent tensor (B, C, T, H, W).
            encoder_hidden_states: CLIP embeddings (B, 1, D).
            motion_hidden_states: Accumulated motion features (B, T, D1, D2).
            motion: New face crop tensor (B, C, T, H, W).
            pose_cond_fea: Accumulated pose features (B, C, T, H, W).
            pose: New pose keypoints (B, C, T, H, W).
            new_noise: Noise for next iteration (B, C, T, H, W).
            d00-u32: Reference hidden states from reference UNet.

        Returns:
            Tuple of:
                - pred_video: Decoded video frames (T, H, W, C).
                - latents: Updated latent tensor for next iteration.
                - pose_cond_fea_out: Updated pose features.
                - motion_hidden_states_out: Updated motion features.
                - motion_out: Motion features for keyframe tracking.
                - latent_first: First latent for potential keyframe update.
        """
        # Encode new pose features
        new_pose_cond_fea = self.pose_guider(pose)
        pose_cond_fea = torch.cat([pose_cond_fea, new_pose_cond_fea], dim=2)

        # Encode new motion features
        new_motion_hidden_states = self.motion_encoder(motion)
        motion_hidden_states = torch.cat([motion_hidden_states, new_motion_hidden_states], dim=1)

        # Prepare encoder hidden states for UNet
        encoder_hidden_states_combined = [encoder_hidden_states, motion_hidden_states]

        # Run UNet with explicit reference hidden states
        score = self.unet(
            sample,
            self.timesteps,
            encoder_hidden_states_combined,
            pose_cond_fea,
            d00, d01, d10, d11, d20, d21, m,
            u10, u11, u12, u20, u21, u22, u30, u31, u32,
        )

        # Scheduler step
        score = rearrange(score, 'b c f h w -> (b f) c h w')
        sample_flat = rearrange(sample, 'b c f h w -> (b f) c h w')

        latents_model_input, pred_original_sample = self.scheduler.step(
            score, self.timesteps, sample_flat, return_dict=False
        )

        latents_model_input = latents_model_input.to(sample.dtype)
        pred_original_sample = pred_original_sample.to(sample.dtype)

        # Reshape back to 5D
        latents_model_input = rearrange(latents_model_input, '(b f) c h w -> b c f h w', f=16)

        # Decode first 4 frames (temporal_window_size)
        pred_video = self._decode_latents(pred_original_sample[:4])

        # Prepare outputs for next iteration
        # Shift latents: drop first temporal_window_size, add new_noise
        latents = torch.cat([latents_model_input[:, :, 4:, :, :], new_noise], dim=2)

        # Shift pose and motion features
        pose_cond_fea_out = pose_cond_fea[:, :, 4:, :, :]
        motion_hidden_states_out = motion_hidden_states[:, 4:, :, :]

        # Motion for keyframe tracking (first frame's motion)
        motion_out = motion_hidden_states[:, :1, :, :]

        # First predicted latent for potential keyframe reference update
        latent_first = pred_original_sample[:1]

        return (
            pred_video,
            latents,
            pose_cond_fea_out,
            motion_hidden_states_out,
            motion_out,
            latent_first,
        )

    def get_sample_input(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Generate sample inputs for ONNX export.

        Args:
            batch_size: Batch size (usually 1).
            height: Output height.
            width: Output width.
            dtype: Data type (usually float16).
            device: Target device.

        Returns:
            Dictionary of sample input tensors.
        """
        # Constants
        tw = 4  # temporal_window_size
        ts = 4  # temporal_adaptive_step
        tb = tw * ts  # temporal batch size (16)
        ml, mc = 32, 16  # motion latent size, channels
        mh, mw = 224, 224  # motion input size

        lh, lw = height // 8, width // 8  # latent height/width

        # UNet channels
        cd0, cd1, cd2, cm = 320, 640, 1280, 1280
        cu1, cu2, cu3 = 1280, 640, 320

        emb = 768  # CLIP embedding dim
        lc, ic = 4, 3  # latent channels, image channels

        shapes = {
            "sample": (batch_size, lc, tb, lh, lw),
            "encoder_hidden_states": (batch_size, 1, emb),
            "motion_hidden_states": (batch_size, tw * (ts - 1), ml, mc),
            "motion": (batch_size, ic, tw, mh, mw),
            "pose_cond_fea": (batch_size, cd0, tw * (ts - 1), lh, lw),
            "pose": (batch_size, ic, tw, height, width),
            "new_noise": (batch_size, lc, tw, lh, lw),
            "d00": (batch_size, lh * lw, cd0),
            "d01": (batch_size, lh * lw, cd0),
            "d10": (batch_size, lh * lw // 4, cd1),
            "d11": (batch_size, lh * lw // 4, cd1),
            "d20": (batch_size, lh * lw // 16, cd2),
            "d21": (batch_size, lh * lw // 16, cd2),
            "m": (batch_size, lh * lw // 64, cm),
            "u10": (batch_size, lh * lw // 16, cu1),
            "u11": (batch_size, lh * lw // 16, cu1),
            "u12": (batch_size, lh * lw // 16, cu1),
            "u20": (batch_size, lh * lw // 4, cu2),
            "u21": (batch_size, lh * lw // 4, cu2),
            "u22": (batch_size, lh * lw // 4, cu2),
            "u30": (batch_size, lh * lw, cu3),
            "u31": (batch_size, lh * lw, cu3),
            "u32": (batch_size, lh * lw, cu3),
        }

        return {name: torch.randn(shape, dtype=dtype, device=device) for name, shape in shapes.items()}

    @staticmethod
    def get_input_names() -> list[str]:
        """Get list of input tensor names for ONNX export."""
        return [
            "sample",
            "encoder_hidden_states",
            "motion_hidden_states",
            "motion",
            "pose_cond_fea",
            "pose",
            "new_noise",
            "d00", "d01", "d10", "d11", "d20", "d21", "m",
            "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32",
        ]

    @staticmethod
    def get_output_names() -> list[str]:
        """Get list of output tensor names for ONNX export."""
        return [
            "pred_video",
            "latents",
            "pose_cond_fea_out",
            "motion_hidden_states_out",
            "motion_out",
            "latent_first",
        ]

    @staticmethod
    def get_dynamic_axes() -> dict[str, dict[int, str]]:
        """Get dynamic axis specifications for ONNX export.

        Returns:
            Dictionary mapping tensor names to their dynamic axes.
        """
        return {
            # Resolution-dependent axes
            "sample": {3: "h_64", 4: "w_64"},
            "pose_cond_fea": {3: "h_64", 4: "w_64"},
            "pose": {3: "h_512", 4: "w_512"},
            "new_noise": {3: "h_64", 4: "w_64"},
            # Dynamic reference hidden states (for keyframe accumulation)
            "d00": {1: "len_4096"},
            "d01": {1: "len_4096"},
            "u30": {1: "len_4096"},
            "u31": {1: "len_4096"},
            "u32": {1: "len_4096"},
            "d10": {1: "len_1024"},
            "d11": {1: "len_1024"},
            "u20": {1: "len_1024"},
            "u21": {1: "len_1024"},
            "u22": {1: "len_1024"},
            "d20": {1: "len_256"},
            "d21": {1: "len_256"},
            "u10": {1: "len_256"},
            "u11": {1: "len_256"},
            "u12": {1: "len_256"},
            "m": {1: "len_64"},
        }
