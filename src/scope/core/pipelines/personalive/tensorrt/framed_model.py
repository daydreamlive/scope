"""Bundled model for TensorRT export.

This module provides the UNetWork class that bundles multiple PersonaLive models
into a single module for ONNX export and TensorRT conversion.

Based on PersonaLive official implementation:
PersonaLive/src/modeling/framed_models.py
"""

import torch
from torch import nn
from einops import rearrange

try:
    from polygraphy.backend.trt import Profile
    POLYGRAPHY_AVAILABLE = True
except ImportError:
    POLYGRAPHY_AVAILABLE = False


class UNetWork(nn.Module):
    """Bundled model for TensorRT export.

    This class wraps the pose_guider, motion_encoder, denoising_unet (explicit reference),
    vae decoder, and scheduler into a single module for ONNX/TensorRT export.

    The forward pass handles a complete denoising step including:
    1. Encoding new pose features
    2. Encoding new motion features
    3. Running the UNet denoiser with explicit reference injection
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
            denoising_unet: UNet3DConditionModelExplicit for denoising.
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

    def decode_slice(self, vae, x):
        """Decode latents to images using VAE."""
        x = x / 0.18215
        x = vae.decode(x).sample
        x = rearrange(x, "b c h w -> b h w c")
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def forward(
        self,
        sample: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        motion_hidden_states: torch.Tensor,
        motion: torch.Tensor,
        pose_cond_fea: torch.Tensor,
        pose: torch.Tensor,
        new_noise: torch.Tensor,
        # Explicit reference hidden states
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
        """Forward pass for TensorRT inference.

        Args:
            sample: Noisy latent (B, 4, 16, H/8, W/8).
            encoder_hidden_states: CLIP embeddings (B, 1, 768).
            motion_hidden_states: Accumulated motion features (B, 12, 32, 16).
            motion: New motion face crops (B, 3, 4, 224, 224).
            pose_cond_fea: Accumulated pose features (B, 320, 12, H/8, W/8).
            pose: New pose keypoints (B, 3, 4, H, W).
            new_noise: Noise for next iteration (B, 4, 4, H/8, W/8).
            d00-u32: Reference hidden states from reference UNet.

        Returns:
            Tuple of:
                - pred_video: Decoded video frames (4, H, W, 3).
                - latents: Updated latents for next iteration.
                - pose_cond_fea_out: Updated pose features.
                - motion_hidden_states_out: Updated motion features.
                - motion_out: First motion frame for keyframe detection.
                - latent_first: First frame latent for potential keyframe update.
        """
        # Encode new pose features and concatenate
        new_pose_cond_fea = self.pose_guider(pose)
        pose_cond_fea = torch.cat([pose_cond_fea, new_pose_cond_fea], dim=2)

        # Encode new motion features and concatenate
        new_motion_hidden_states = self.motion_encoder(motion)
        motion_hidden_states = torch.cat([motion_hidden_states, new_motion_hidden_states], dim=1)

        # Prepare encoder hidden states for UNet [clip_embeds, motion_features]
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
        pred_video = self.decode_slice(self.vae, pred_original_sample[:4])

        # Prepare outputs for next iteration
        # Shift latents: drop first temporal_window_size, add new_noise
        latents = torch.cat([latents_model_input[:, :, 4:, :, :], new_noise], dim=2)

        # Shift pose and motion features
        pose_cond_fea_out = pose_cond_fea[:, :, 4:, :, :]
        motion_hidden_states_out = motion_hidden_states[:, 4:, :, :]

        # First motion frame for keyframe tracking
        motion_out = motion_hidden_states[:, :1, :, :]

        # First frame latent for potential keyframe update
        latent_first = pred_original_sample[:1]

        return pred_video, latents, pose_cond_fea_out, motion_hidden_states_out, motion_out, latent_first

    def get_sample_input(self, batchsize: int, height: int, width: int, dtype, device):
        """Generate sample inputs for ONNX export.

        Args:
            batchsize: Batch size (typically 1).
            height: Output image height.
            width: Output image width.
            dtype: Tensor dtype.
            device: Target device.

        Returns:
            Dictionary of sample tensors for ONNX export.
        """
        tw, ts, tb = 4, 4, 16  # temporal_window_size, temporal_adaptive_steps, temporal_batch
        ml, mc, mh, mw = 32, 16, 224, 224  # motion latent dims
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8  # latent height/width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320  # unet channels
        emb = 768  # CLIP embedding dims
        lc, ic = 4, 3  # latent/image channels

        profile = {
            "sample": [b, lc, tb, lh, lw],
            "encoder_hidden_states": [b, 1, emb],
            "motion_hidden_states": [b, tw * (ts - 1), ml, mc],
            "motion": [b, ic, tw, mh, mw],
            "pose_cond_fea": [b, cd0, tw * (ts - 1), lh, lw],
            "pose": [b, ic, tw, h, w],
            "new_noise": [b, lc, tw, lh, lw],
            "d00": [b, lh * lw, cd0],
            "d01": [b, lh * lw, cd0],
            "d10": [b, lh * lw // 4, cd1],
            "d11": [b, lh * lw // 4, cd1],
            "d20": [b, lh * lw // 16, cd2],
            "d21": [b, lh * lw // 16, cd2],
            "m": [b, lh * lw // 64, cm],
            "u10": [b, lh * lw // 16, cu1],
            "u11": [b, lh * lw // 16, cu1],
            "u12": [b, lh * lw // 16, cu1],
            "u20": [b, lh * lw // 4, cu2],
            "u21": [b, lh * lw // 4, cu2],
            "u22": [b, lh * lw // 4, cu2],
            "u30": [b, lh * lw, cu3],
            "u31": [b, lh * lw, cu3],
            "u32": [b, lh * lw, cu3],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}

    @staticmethod
    def get_input_names():
        """Get ordered input names for ONNX export."""
        return [
            "sample", "encoder_hidden_states", "motion_hidden_states",
            "motion", "pose_cond_fea", "pose", "new_noise",
            "d00", "d01", "d10", "d11", "d20", "d21", "m",
            "u10", "u11", "u12", "u20", "u21", "u22", "u30", "u31", "u32"
        ]

    @staticmethod
    def get_output_names():
        """Get ordered output names for ONNX export."""
        return [
            "pred_video", "latents", "pose_cond_fea_out",
            "motion_hidden_states_out", "motion_out", "latent_first"
        ]

    @staticmethod
    def get_dynamic_axes():
        """Get dynamic axes for ONNX export."""
        return {
            "sample": {3: "h_64", 4: "w_64"},
            "pose_cond_fea": {3: "h_64", 4: "w_64"},
            "pose": {3: "h_512", 4: "w_512"},
            "new_noise": {3: "h_64", 4: "w_64"},
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

    def get_dynamic_profile(self, batchsize: int, height: int, width: int):
        """Get TensorRT optimization profile with dynamic shapes.

        Args:
            batchsize: Batch size.
            height: Target image height.
            width: Target image width.

        Returns:
            Polygraphy Profile object with min/opt/max shapes.
        """
        if not POLYGRAPHY_AVAILABLE:
            raise ImportError("polygraphy is required for TensorRT profiles")

        tw, ts, tb = 4, 4, 16
        ml, mc, mh, mw = 32, 16, 224, 224
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320
        emb = 768
        lc, ic = 4, 3

        # Fixed inputs (don't change with keyframes)
        fixed_inputs_map = {
            "sample": (b, lc, tb, lh, lw),
            "encoder_hidden_states": (b, 1, emb),
            "motion_hidden_states": (b, tw * (ts - 1), ml, mc),
            "motion": (b, ic, tw, mh, mw),
            "pose_cond_fea": (b, cd0, tw * (ts - 1), lh, lw),
            "pose": (b, ic, tw, h, w),
            "new_noise": (b, lc, tw, lh, lw),
        }

        # Dynamic inputs (grow with keyframe accumulation: 1x, 2x, 4x)
        dynamic_inputs_map = {
            "d00": (b, lh * lw, cd0),
            "d01": (b, lh * lw, cd0),
            "d10": (b, lh * lw // 4, cd1),
            "d11": (b, lh * lw // 4, cd1),
            "d20": (b, lh * lw // 16, cd2),
            "d21": (b, lh * lw // 16, cd2),
            "m": (b, lh * lw // 64, cm),
            "u10": (b, lh * lw // 16, cu1),
            "u11": (b, lh * lw // 16, cu1),
            "u12": (b, lh * lw // 16, cu1),
            "u20": (b, lh * lw // 4, cu2),
            "u21": (b, lh * lw // 4, cu2),
            "u22": (b, lh * lw // 4, cu2),
            "u30": (b, lh * lw, cu3),
            "u31": (b, lh * lw, cu3),
            "u32": (b, lh * lw, cu3),
        }

        profile = Profile()

        # Fixed inputs have same min/opt/max
        for name, shape in fixed_inputs_map.items():
            shape_tuple = tuple(shape)
            profile.add(name, min=shape_tuple, opt=shape_tuple, max=shape_tuple)

        # Dynamic inputs can grow 1x-4x in the sequence dimension
        for name, base_shape in dynamic_inputs_map.items():
            dim0, dim1_base, dim2 = base_shape

            val_1x = dim1_base * 1  # 1 keyframe
            val_2x = dim1_base * 2  # 2 keyframes
            val_4x = dim1_base * 4  # 4 keyframes (max)

            min_shape = (dim0, val_1x, dim2)
            opt_shape = (dim0, val_2x, dim2)
            max_shape = (dim0, val_4x, dim2)

            profile.add(name, min=min_shape, opt=opt_shape, max=max_shape)

        return profile
