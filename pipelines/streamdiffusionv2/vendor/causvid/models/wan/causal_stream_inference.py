from .. import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
)
from typing import List, Optional
import torch
import time
import os
import logging


logger = logging.getLogger(__name__)


class CausalStreamInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, enable_i2v: bool = False):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(args, "generator_name", args.model_name)

        model_dir = getattr(args, "model_dir", None)
        text_encoder_path = getattr(args, "text_encoder_path", None)
        tokenizer_path = getattr(args, "tokenizer_path", None)

        start = time.time()
        self.generator = get_diffusion_wrapper(model_name=self.generator_model_name)(
            model_dir=model_dir
        )
        print(f"Loaded diffusion wrapper in {time.time() - start:3f}s")

        start = time.time()
        self.text_encoder = get_text_encoder_wrapper(model_name=args.model_name)(
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")

        start = time.time()
        self.vae = get_vae_wrapper(model_name=args.model_name)(model_dir=model_dir)
        print(f"Loaded VAE in {time.time() - start:3f}s")

        self.enable_i2v = enable_i2v

        # Add CLIP encoder if I2V mode is enabled
        if enable_i2v:
            from .wan_base.modules.clip import CLIPModel

            # Full CLIP model from DeepBeepMeep/Wan2.1
            clip_checkpoint = os.path.join(
                model_dir, "Wan2.1-T2V-1.3B", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            )
            # Tokenizer directory path (downloaded from DeepBeepMeep/Wan2.1/xlm-roberta-large)
            clip_tokenizer = os.path.join(
                model_dir, "Wan2.1-T2V-1.3B", "xlm-roberta-large"
            )

            if os.path.exists(clip_checkpoint):
                start = time.time()
                self.clip = CLIPModel(
                    dtype=torch.bfloat16,  # Match VAE/Generator dtype
                    device=torch.device("cpu"),  # Start on CPU to save VRAM
                    checkpoint_path=clip_checkpoint,
                    tokenizer_path=clip_tokenizer,
                )
                print(f"Loaded CLIP encoder for I2V in {time.time() - start:.3f}s")
                logger.info("CLIP encoder initialized for I2V mode")
            else:
                logger.warning(f"CLIP checkpoint not found at {clip_checkpoint}. I2V mode will not work.")
                logger.warning("Please download CLIP model: python download_models.py --pipeline streamdiffusionv2")
                self.clip = None
        else:
            self.clip = None

        # Step 2: Initialize all causal hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device
        )
        assert self.denoising_step_list[-1] == 0
        # remove the last timestep (which equals zero)
        self.denoising_step_list = self.denoising_step_list[:-1]

        self.scheduler = self.generator.get_scheduler()
        if (
            args.warp_denoising_step
        ):  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            ).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        scale_size = 16
        self.frame_seq_length = (args.height // scale_size) * (args.width // scale_size)
        self.kv_cache_length = self.frame_seq_length * args.num_kv_cache

        self.conditional_dict = None

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, self.kv_cache_length, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, self.kv_cache_length, 12, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
            )

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, 12, 128], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            )

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def prepare(self, noise: torch.Tensor, text_prompts: List[str]):
        batch_size = noise.shape[0]
        self.conditional_dict = self.text_encoder(text_prompts=text_prompts)
        if batch_size > 1:
            self.conditional_dict["prompt_embeds"] = self.conditional_dict[
                "prompt_embeds"
            ].repeat(batch_size, 1, 1)

        # Step 1: Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )

            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False

    def inference(
        self,
        noise: torch.Tensor,
        current_start: int,
        current_end: int,
        current_step: int,
        generator: Optional[torch.Generator] = None,
        visual_context: Optional[torch.Tensor] = None,
        cond_concat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = noise.shape[0]

        # Step 2.1: Spatial denoising loop
        self.denoising_step_list[0] = current_step
        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = (
                torch.ones(
                    [batch_size, noise.shape[1]], device=noise.device, dtype=torch.int64
                )
                * current_timestep
            )

            if index < len(self.denoising_step_list) - 1:
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    visual_context=visual_context,
                    cond_concat=cond_concat,
                )
                next_timestep = self.denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=generator,
                )
                noise = self.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones([batch_size], device="cuda", dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                # for getting real output
                denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                    visual_context=visual_context,
                    cond_concat=cond_concat,
                )

        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=self.conditional_dict,
            timestep=timestep * 0,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            visual_context=visual_context,
            cond_concat=cond_concat,
        )

        return denoised_pred

    def encode_image_for_i2v(self, image_tensor: torch.Tensor, num_latent_frames: int = 2):
        """
        Encode image for I2V conditioning.

        Args:
            image_tensor: Preprocessed image [3, H, W] in [-1, 1] range
            num_latent_frames: Number of temporal latent frames (default 2 for start_chunk_size=5)

        Returns:
            visual_context: CLIP features [1, 257, 1280]
            cond_concat: Conditioning with mask [T, 17, H/8, W/8] in FCHW format
                        (T=num frames, 17=1 mask + 16 latent channels)
        """
        if not self.enable_i2v or self.clip is None:
            raise RuntimeError("I2V mode not enabled or CLIP not available")

        device = next(self.generator.model.parameters()).device
        h, w = image_tensor.shape[1:]
        lat_h, lat_w = h // 8, w // 8

        # Move image to device
        image_seq = image_tensor[:, None, :, :].to(device, dtype=torch.bfloat16)

        # CLIP encode for visual context
        # Add single frame dimension for CLIP: [3, H, W] -> [3, 1, H, W]
        self.clip.model.to(device)
        with torch.no_grad():
            visual_context = self.clip.visual([image_seq])  # [1, 257, 1280]

        # Offload CLIP to save VRAM
        self.clip.model.cpu()

        # VAE encode the single image frame
        # Convert to BCTHW format: [3, 1, H, W] -> [1, 3, 1, H, W]
        image_bcthw = image_seq.unsqueeze(0)  # [1, 3, 1, H, W]

        with torch.no_grad():
            # Encode single frame to latent space
            cond_latent = self.vae.model.encode(
                image_bcthw,
                scale=[self.vae.mean.to(device), 1.0 / self.vae.std.to(device)]
            )  # [1, 16, 1, H/8, W/8]

        # Expand cond_latent to match temporal dimension
        # [1, 16, 1, H/8, W/8] -> [1, 16, T, H/8, W/8]
        cond_latent_expanded = torch.zeros(
            1, 16, num_latent_frames, lat_h, lat_w,
            device=device, dtype=cond_latent.dtype
        )
        cond_latent_expanded[:, :, 0, :, :] = cond_latent[:, :, 0, :, :]

        # Create mask: [1, 1, T, H/8, W/8]
        # 1 for first temporal frame (conditioned), 0 for rest (to be generated)
        mask = torch.zeros(1, 1, num_latent_frames, lat_h, lat_w, device=device, dtype=cond_latent.dtype)
        mask[:, :, 0, :, :] = 1.0

        # Concatenate mask and latent along channel dimension
        # [1, 1, T, H/8, W/8] + [1, 16, T, H/8, W/8] -> [1, 17, T, H/8, W/8]
        cond_concat = torch.cat([mask, cond_latent_expanded], dim=1)

        # Convert from BCFHW [1, 17, T, H/8, W/8] to FCHW [T, 17, H/8, W/8] format
        # The wrapper permutes noise to BFCHW, so after batch iteration we get FCHW
        # Our conditioning must match this FCHW format for concatenation to work
        cond_concat = cond_concat.squeeze(0)  # Remove batch: [17, T, H/8, W/8] CFHW
        cond_concat = cond_concat.permute(1, 0, 2, 3)  # Permute to FCHW: [T, 17, H/8, W/8]

        logger.info(f"Encoded image for I2V: visual_context={visual_context.shape}, cond_concat={cond_concat.shape}")

        return visual_context, cond_concat
