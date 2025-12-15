"""PersonaLive pipeline for real-time portrait animation.

Animates a reference portrait image using driving video frames.
Based on: https://github.com/GVCLab/PersonaLive
"""

import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..interface import Pipeline, Requirements
from ..process import preprocess_chunk, postprocess_chunk
from ..schema import PersonaLiveConfig
from ..utils import load_model_config
from .components import FaceDetector
from .modules import (
    DDIMScheduler,
    MotEncoder,
    MotionExtractor,
    PoseGuider,
    ReferenceAttentionControl,
    UNet2DConditionModel,
    UNet3DConditionModel,
    draw_keypoints,
    get_boxes,
)

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class PersonaLivePipeline(Pipeline):
    """PersonaLive portrait animation pipeline.

    This pipeline animates a reference portrait image using driving video frames.
    It requires:
    1. A reference image to be set once via `fuse_reference()`
    2. Continuous driving video frames via `__call__()`

    The output is animated video frames of the reference portrait following
    the motion from the driving video.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return PersonaLiveConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize PersonaLive pipeline.

        Args:
            config: Pipeline configuration (OmegaConf or dict-like).
            device: Target device for models.
            dtype: Data type for model weights.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.numpy_dtype = np.float16 if dtype == torch.float16 else np.float32

        # Load model config
        model_config = load_model_config(config, __file__)

        # Get paths from config
        model_dir = Path(getattr(config, "model_dir", "."))
        personalive_dir = model_dir / "PersonaLive" / "pretrained_weights"

        # Model paths
        pretrained_base_path = personalive_dir / "sd-image-variations-diffusers"
        vae_path = personalive_dir / "sd-vae-ft-mse"
        personalive_weights = personalive_dir / "personalive"
        image_encoder_path = pretrained_base_path / "image_encoder"

        # Pipeline parameters
        self.temporal_window_size = getattr(model_config, "temporal_window_size", 4)
        self.temporal_adaptive_step = getattr(model_config, "temporal_adaptive_step", 4)
        self.num_inference_steps = getattr(model_config, "num_inference_steps", 4)
        self.batch_size = getattr(model_config, "batch_size", 1)

        # Resolution
        self.height = getattr(config, "height", 512)
        self.width = getattr(config, "width", 512)
        self.ref_height = getattr(model_config, "reference_image_height", 512)
        self.ref_width = getattr(model_config, "reference_image_width", 512)

        # Load inference config for UNet kwargs
        unet_additional_kwargs = getattr(model_config, "unet_additional_kwargs", {})
        if isinstance(unet_additional_kwargs, OmegaConf):
            unet_additional_kwargs = OmegaConf.to_container(unet_additional_kwargs)

        # Scheduler config
        sched_kwargs = getattr(model_config, "noise_scheduler_kwargs", {})
        if isinstance(sched_kwargs, OmegaConf):
            sched_kwargs = OmegaConf.to_container(sched_kwargs)

        logger.info("Loading PersonaLive models...")
        start = time.time()

        # Load pose guider
        self.pose_guider = PoseGuider().to(device=device, dtype=dtype)
        pose_guider_path = personalive_weights / "pose_guider.pth"
        if pose_guider_path.exists():
            state_dict = torch.load(pose_guider_path, map_location="cpu")
            self.pose_guider.load_state_dict(state_dict)
            del state_dict
        logger.info(f"Loaded pose_guider in {time.time() - start:.2f}s")

        # Load motion encoder
        start = time.time()
        self.motion_encoder = MotEncoder().to(dtype=dtype, device=device).eval()
        motion_encoder_path = personalive_weights / "motion_encoder.pth"
        if motion_encoder_path.exists():
            state_dict = torch.load(motion_encoder_path, map_location="cpu")
            self.motion_encoder.load_state_dict(state_dict)
            del state_dict
        logger.info(f"Loaded motion_encoder in {time.time() - start:.2f}s")

        # Load pose encoder (motion extractor)
        start = time.time()
        self.pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=dtype).eval()
        pose_encoder_path = personalive_weights / "motion_extractor.pth"
        if pose_encoder_path.exists():
            state_dict = torch.load(pose_encoder_path, map_location="cpu")
            self.pose_encoder.load_state_dict(state_dict, strict=False)
            del state_dict
        logger.info(f"Loaded pose_encoder in {time.time() - start:.2f}s")

        # Load denoising UNet (3D)
        start = time.time()
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            str(pretrained_base_path),
            "",
            subfolder="unet",
            unet_additional_kwargs=unet_additional_kwargs,
        ).to(dtype=dtype, device=device)

        denoising_unet_path = personalive_weights / "denoising_unet.pth"
        if denoising_unet_path.exists():
            state_dict = torch.load(denoising_unet_path, map_location="cpu")
            self.denoising_unet.load_state_dict(state_dict, strict=False)
            del state_dict

        temporal_module_path = personalive_weights / "temporal_module.pth"
        if temporal_module_path.exists():
            state_dict = torch.load(temporal_module_path, map_location="cpu")
            self.denoising_unet.load_state_dict(state_dict, strict=False)
            del state_dict
        logger.info(f"Loaded denoising_unet in {time.time() - start:.2f}s")

        # Load reference UNet (2D)
        start = time.time()
        self.reference_unet = UNet2DConditionModel.from_pretrained(
            str(pretrained_base_path),
            subfolder="unet",
        ).to(dtype=dtype, device=device)

        reference_unet_path = personalive_weights / "reference_unet.pth"
        if reference_unet_path.exists():
            state_dict = torch.load(reference_unet_path, map_location="cpu")
            self.reference_unet.load_state_dict(state_dict)
            del state_dict
        logger.info(f"Loaded reference_unet in {time.time() - start:.2f}s")

        # Setup reference attention control
        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            batch_size=self.batch_size,
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            batch_size=self.batch_size,
            fusion_blocks="full",
            cache_kv=True,
        )

        # Load VAE
        start = time.time()
        self.vae = AutoencoderKL.from_pretrained(str(vae_path)).to(
            device=device, dtype=dtype
        )
        logger.info(f"Loaded VAE in {time.time() - start:.2f}s")

        # Load image encoder
        start = time.time()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(image_encoder_path),
        ).to(device=device, dtype=dtype)
        logger.info(f"Loaded image_encoder in {time.time() - start:.2f}s")

        # Setup scheduler
        self.scheduler = DDIMScheduler(**sched_kwargs)
        self.timesteps = torch.tensor([999, 666, 333, 0], device=device).long()
        self.scheduler.set_step_length(333)

        # Setup generator
        seed = getattr(config, "seed", 42)
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)

        # Image processors
        self.vae_scale_factor = 8
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=True
        )

        # Face detector
        self.face_detector = FaceDetector()

        # State variables
        self.first_frame = True
        self.motion_bank = None
        self.count = 0
        self.num_khf = 0  # Number of history keyframes

        # Temporal buffers
        self.latents_pile = deque([])
        self.pose_pile = deque([])
        self.motion_pile = deque([])

        # Reference state (set via fuse_reference)
        self.reference_fused = False
        self.encoder_hidden_states = None
        self.ref_image_tensor = None
        self.ref_image_latents = None
        self.ref_cond_tensor = None
        self.kps_ref = None
        self.kps_frame1 = None

        # Enable memory efficient attention
        self._enable_xformers()

        torch.cuda.empty_cache()
        logger.info("PersonaLive pipeline initialized")

    def _enable_xformers(self):
        """Enable xformers memory efficient attention if available."""
        try:
            self.reference_unet.enable_xformers_memory_efficient_attention()
            self.denoising_unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements.

        PersonaLive requires:
        - Reference image to be fused first via fuse_reference()
        - Driving video frames (temporal_window_size frames per call)

        Returns Requirements only when video mode is signaled (video key in kwargs),
        following the same pattern as other video pipelines.
        """
        # Only return requirements when video mode is signaled
        # This is indicated by FrameProcessor setting video=True in prepare_params
        if kwargs.get("video") is not None:
            return Requirements(input_size=self.temporal_window_size)
        return None

    def _fast_resize(self, images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Fast bilinear resize of image tensor."""
        return F.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def _interpolate_tensors(self, a: torch.Tensor, b: torch.Tensor, num: int) -> torch.Tensor:
        """Linear interpolation between tensors a and b."""
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

        B, _, *rest = a.shape
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (1, num) + (1,) * len(rest)
        alphas = alphas.view(view_shape)

        result = (1 - alphas) * a + alphas * b
        return result

    def _calculate_dis(self, A: torch.Tensor, B: torch.Tensor, threshold: float = 10.0):
        """Calculate distance between motion features for keyframe selection."""
        A_flat = A.view(A.size(1), -1).clone()
        B_flat = B.view(B.size(1), -1).clone()

        dist = torch.cdist(B_flat.to(torch.float32), A_flat.to(torch.float32), p=2)
        min_dist, min_idx = dist.min(dim=1)

        idx_to_add = torch.nonzero(min_dist[:1] > threshold, as_tuple=False).squeeze(1).tolist()

        if len(idx_to_add) > 0:
            B_to_add = B[:, idx_to_add]
            A_new = torch.cat([A, B_to_add], dim=1)
        else:
            A_new = A

        return idx_to_add, A_new, min_idx

    def _crop_face_tensor(self, image_tensor: torch.Tensor, boxes) -> torch.Tensor:
        """Crop face from tensor using bounding box."""
        left, top, right, bot = boxes
        left, top, right, bottom = map(int, (left, top, right, bot))

        face_patch = image_tensor[:, top:bottom, left:right]
        face_patch = F.interpolate(
            face_patch.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        return face_patch

    @torch.no_grad()
    def fuse_reference(self, ref_image: Image.Image):
        """Fuse reference image into the pipeline.

        This must be called before processing driving frames.

        Args:
            ref_image: PIL Image of the reference portrait.
        """
        logger.info("Fusing reference image...")

        # Process for CLIP - resize to 224x224 first (matching official implementation)
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values

        # Process for VAE
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=self.ref_height, width=self.ref_width
        )

        # Get CLIP embeddings
        clip_image_embeds = self.image_encoder(
            clip_image.to(self.image_encoder.device, dtype=self.image_encoder.dtype)
        ).image_embeds
        self.encoder_hidden_states = clip_image_embeds.unsqueeze(1)

        # Encode reference image
        ref_image_tensor = ref_image_tensor.to(dtype=self.vae.dtype, device=self.vae.device)
        self.ref_image_tensor = ref_image_tensor.squeeze(0)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215

        # Run reference UNet to cache features
        self.reference_unet(
            ref_image_latents.to(self.reference_unet.device),
            torch.zeros((self.batch_size,), dtype=self.dtype, device=self.reference_unet.device),
            encoder_hidden_states=self.encoder_hidden_states,
            return_dict=False,
        )
        self.reference_control_reader.update(self.reference_control_writer)
        self.encoder_hidden_states = self.encoder_hidden_states.to(self.device)

        # Prepare conditioning tensor for pose encoder
        ref_cond_tensor = self.cond_image_processor.preprocess(
            ref_image, height=256, width=256
        ).to(device=self.device, dtype=self.pose_encoder.dtype)
        self.ref_cond_tensor = ref_cond_tensor / 2 + 0.5
        self.ref_image_latents = ref_image_latents

        # Reset state - clear ALL piles first to handle re-fusing
        self.first_frame = True
        self.motion_bank = None
        self.count = 0
        self.num_khf = 0
        self.latents_pile.clear()
        self.pose_pile.clear()
        self.motion_pile.clear()

        # Initialize latent piles with padding
        padding_num = (self.temporal_adaptive_step - 1) * self.temporal_window_size
        init_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, padding_num, 1, 1)
        noise = torch.randn_like(init_latents)
        init_timesteps = reversed(self.timesteps).repeat_interleave(self.temporal_window_size, dim=0)
        noisy_latents_first = self.scheduler.add_noise(init_latents, noise, init_timesteps[:padding_num])

        for i in range(self.temporal_adaptive_step - 1):
            l = i * self.temporal_window_size
            r = (i + 1) * self.temporal_window_size
            self.latents_pile.append(noisy_latents_first[:, :, l:r])

        self.reference_fused = True
        logger.info("Reference image fused successfully")

    @torch.no_grad()
    def __call__(
        self,
        video: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process driving video frames.

        Args:
            video: Driving video frames. Can be:
                - Tensor of shape (B, C, T, H, W) in [0, 1] range
                - List of frame tensors in THWC format and [0, 255] range
                - None (not supported)
            **kwargs: Additional parameters (ignored).

        Returns:
            Animated output frames in THWC format and [0, 1] range.
        """
        if not self.reference_fused:
            raise RuntimeError(
                "Reference image must be fused before processing. "
                "Call fuse_reference() first."
            )

        # Convert input to expected format
        if video is None:
            # No video frames yet - return empty tensor
            # This can happen during streaming startup
            logger.debug("No video frames provided, returning empty output")
            return torch.empty(0, self.height, self.width, 3, device="cpu")

        # Use standard preprocess_chunk for list input (from frame_processor)
        if isinstance(video, list):
            # preprocess_chunk expects list of (1, H, W, C) tensors
            # Returns (B, C, T, H, W) tensor in [-1, 1] range
            frames = preprocess_chunk(
                video,
                self.device,
                self.dtype,
                height=self.height,
                width=self.width,
            )
        else:
            # Already a tensor - ensure correct device/dtype
            frames = video.to(device=self.device, dtype=self.dtype)

        # Process input frames
        output = self._process_frames(frames)

        return output

    @torch.no_grad()
    def _process_frames(self, images: torch.Tensor) -> torch.Tensor:
        """Process driving frames through the pipeline.

        Args:
            images: Input tensor of shape (B, C, T, H, W) in [-1, 1] or [0, 1] range.

        Returns:
            Output frames tensor of shape (T, H, W, C) in [0, 1] range.
        """
        batch_size = self.batch_size
        device = self.device
        temporal_window_size = self.temporal_window_size
        temporal_adaptive_step = self.temporal_adaptive_step

        # Reshape from (B, C, T, H, W) to (T, C, H, W) for processing
        if images.dim() == 5:
            images = images.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)

        # PersonaLive expects exactly temporal_window_size frames per call
        # If we receive more, process only the required number
        num_frames = images.shape[0]
        if num_frames != temporal_window_size:
            logger.warning(
                f"Expected {temporal_window_size} frames but received {num_frames}. "
                f"Processing only the last {temporal_window_size} frames."
            )
            images = images[-temporal_window_size:]

        # Store original images for face cropping (they should be in [0, 1] after preprocess_chunk)
        # But ensure they're in [0, 1] range for consistency
        if images.min() < 0:
            images = images / 2 + 0.5
        images = images.clamp(0, 1)

        # Resize for pose encoder (matching official: resize first, then normalize)
        # Official does: fast_resize(images, 256, 256) then / 2 + 0.5
        # This suggests images are in [-1, 1] before resize in official
        # But since we have [0, 1], we'll resize directly and ensure [0, 1] range
        tgt_cond_tensor = self._fast_resize(images, 256, 256)
        # Ensure [0, 1] range (official normalizes after resize, but we already have [0, 1])
        tgt_cond_tensor = tgt_cond_tensor.clamp(0, 1)

        # Get keypoints
        if self.first_frame:
            mot_bbox_param, kps_ref, kps_frame1, kps_dri = self.pose_encoder.interpolate_kps_online(
                self.ref_cond_tensor, tgt_cond_tensor, num_interp=12 + 1
            )
            self.kps_ref = kps_ref
            self.kps_frame1 = kps_frame1
        else:
            mot_bbox_param, kps_dri = self.pose_encoder.get_kps(
                self.kps_ref, self.kps_frame1, tgt_cond_tensor
            )

        # Draw keypoints and get bounding boxes
        keypoints = draw_keypoints(mot_bbox_param, device=device)
        boxes = get_boxes(kps_dri)
        keypoints = rearrange(keypoints.unsqueeze(2), 'f c b h w -> b c f h w')
        keypoints = keypoints.to(device=device, dtype=self.pose_guider.dtype)

        # Process motion features
        if self.first_frame:
            ref_box = get_boxes(mot_bbox_param[:1])
            ref_face = self._crop_face_tensor(self.ref_image_tensor, ref_box[0])
            motion_face = [ref_face]
            for i, frame in enumerate(images):
                motion_face.append(self._crop_face_tensor(frame, boxes[i]))
            pose_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
            motion_hidden_states = self.motion_encoder(pose_cond_tensor)
            ref_motion = motion_hidden_states[:, :1]
            dri_motion = motion_hidden_states[:, 1:]

            init_motion_hidden_states = self._interpolate_tensors(
                ref_motion, dri_motion[:, :1], num=12 + 1
            )[:, :-1]
            for i in range(temporal_adaptive_step - 1):
                l = i * temporal_window_size
                r = (i + 1) * temporal_window_size
                self.motion_pile.append(init_motion_hidden_states[:, l:r])
            self.motion_pile.append(dri_motion)

            self.motion_bank = ref_motion
        else:
            motion_face = []
            for i, frame in enumerate(images):
                motion_face.append(self._crop_face_tensor(frame, boxes[i]))
            pose_cond_tensor = torch.cat(motion_face, dim=0).transpose(0, 1)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
            motion_hidden_states = self.motion_encoder(pose_cond_tensor)
            self.motion_pile.append(motion_hidden_states)

        # Encode pose features
        pose_fea = self.pose_guider(keypoints)
        if self.first_frame:
            for i in range(temporal_adaptive_step):
                l = i * temporal_window_size
                r = (i + 1) * temporal_window_size
                self.pose_pile.append(pose_fea[:, :, l:r])
            self.first_frame = False
        else:
            self.pose_pile.append(pose_fea)

        # Prepare noisy latents for new frames
        latents = self.ref_image_latents.unsqueeze(2).repeat(1, 1, temporal_window_size, 1, 1)
        noise = torch.randn_like(latents)
        latents = self.scheduler.add_noise(latents, noise, self.timesteps[:1])
        self.latents_pile.append(latents)

        # Combine piles
        jump = 1
        motion_hidden_state = torch.cat(list(self.motion_pile), dim=1)
        pose_cond_fea = torch.cat(list(self.pose_pile), dim=2)

        # Check for keyframe addition
        idx_to_add = []
        if self.count > 8:
            idx_to_add, self.motion_bank, idx_his = self._calculate_dis(
                self.motion_bank, motion_hidden_state, threshold=17.0
            )

        # Denoise
        latents_model_input = torch.cat(list(self.latents_pile), dim=2)
        for j in range(jump):
            timesteps = reversed(self.timesteps[j::jump]).repeat_interleave(temporal_window_size, dim=0)
            timesteps = torch.stack([timesteps] * batch_size)
            timesteps = rearrange(timesteps, 'b f -> (b f)')

            noise_pred = self.denoising_unet(
                latents_model_input,
                timesteps,
                encoder_hidden_states=[
                    self.encoder_hidden_states,
                    motion_hidden_state,
                ],
                pose_cond_fea=pose_cond_fea,
                return_dict=False,
            )[0]

            clip_length = noise_pred.shape[2]
            mid_noise_pred = rearrange(noise_pred, 'b c f h w -> (b f) c h w')
            mid_latents = rearrange(latents_model_input, 'b c f h w -> (b f) c h w')
            latents_model_input, pred_original_sample = self.scheduler.step(
                mid_noise_pred, timesteps, mid_latents,
                generator=self.generator, return_dict=False
            )
            latents_model_input = rearrange(latents_model_input, '(b f) c h w -> b c f h w', f=clip_length)
            pred_original_sample = rearrange(pred_original_sample, '(b f) c h w -> b c f h w', f=clip_length)
            latents_model_input = torch.cat([
                pred_original_sample[:, :, :temporal_window_size],
                latents_model_input[:, :, temporal_window_size:]
            ], dim=2)
            latents_model_input = latents_model_input.to(dtype=self.dtype)

        # History keyframe mechanism
        if len(idx_to_add) > 0 and self.num_khf < 3:
            self.reference_control_writer.clear()
            self.reference_unet(
                pred_original_sample[:, :, 0].to(self.reference_unet.dtype),
                torch.zeros((batch_size,), dtype=self.dtype, device=self.reference_unet.device),
                encoder_hidden_states=self.encoder_hidden_states,
                return_dict=False,
            )
            self.reference_control_reader.update_hkf(self.reference_control_writer)
            logger.debug("Added history keyframe")
            self.num_khf += 1

        # Update latent piles
        for i in range(len(self.latents_pile)):
            self.latents_pile[i] = latents_model_input[
                :, :, i * temporal_adaptive_step:(i + 1) * temporal_adaptive_step, :, :
            ]

        # Pop oldest latents and decode
        self.pose_pile.popleft()
        self.motion_pile.popleft()
        latents = self.latents_pile.popleft()
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "b c h w -> b h w c")
        video = (video / 2 + 0.5).clamp(0, 1)

        self.count += 1

        # Return in THWC format [0, 1]
        return video.cpu()
