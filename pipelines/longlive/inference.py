# Contains code based on https://github.com/NVlabs/LongLive
import logging

import torch

from ..process import postprocess_chunk

# The VAE compresses a pixel frame into a latent frame which consists of patches
# The patch embedding converts spatial patches into tokens
# The VAE does 8x spatial downsampling
# The patch embedding does 2x spatial downsampling
# Thus, we end up spatially scaling down by 16
SCALE_SIZE = 16

# The VAE does 8x spatial downsampling
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 8

# https://github.com/daydreamlive/scope/blob/a6a7aa1d7a3be60d3b444e254f83a9fd09e9151a/pipelines/base/wan2_1/modules/causal_model.py#L117
MAX_ROPE_FREQ_TABLE_SEQ_LEN = 1024

logger = logging.getLogger(__name__)


class InferencePipeline(torch.nn.Module):
    def __init__(
        self,
        config,
        generator,
        text_encoder,
        vae,
        low_memory: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        # The height and width must be divisible by SCALE_SIZE
        req_height = config.get("height", 512)
        req_width = config.get("width", 512)
        self.height = round(req_height / SCALE_SIZE) * SCALE_SIZE
        self.width = round(req_width / SCALE_SIZE) * SCALE_SIZE

        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae
        self.low_memory = low_memory
        self.seed = seed
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            config.denoising_step_list, dtype=torch.long
        )
        if config.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = (self.height // SCALE_SIZE) * (self.width // SCALE_SIZE)

        self.kv_cache1 = None
        self.config = config
        self.batch_size = 1
        self.local_attn_size = config.model_kwargs.local_attn_size
        self.recache_buffer = None
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 1)

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.conditional_dict = None
        self.current_start = 0

    @torch.no_grad()
    def prepare(
        self,
        prompts: list[str] = None,
        init_cache: bool = False,
    ):
        # CausalWanModel uses a RoPE frequency table with a max sequence length of 1024
        # This means that it has positions for 1024 latent frames
        # current_start is used to index into this table
        # We need to make sure that current_start does not shift past the max sequence length of the RoPE frequency table
        # when slicing here https://github.com/daydreamlive/scope/blob/a6a7aa1d7a3be60d3b444e254f83a9fd09e9151a/pipelines/base/wan2_1/modules/causal_model.py#L52
        # When we hit the limit we reset the caches and indices
        max_current_start = MAX_ROPE_FREQ_TABLE_SEQ_LEN - self.num_frame_per_block
        if self.current_start >= max_current_start:
            init_cache = True

        if prompts is not None:
            # Make sure text encoder is on right device
            generator_device = next(self.generator.model.parameters()).device
            self.text_encoder = self.text_encoder.to(generator_device)

            self.conditional_dict = self.text_encoder(text_prompts=prompts)
            if self.batch_size > 1:
                self.conditional_dict["prompt_embeds"] = self.conditional_dict[
                    "prompt_embeds"
                ].repeat(self.batch_size, 1, 1)

            # If in low memory mode offload text encoder to CPU
            if self.low_memory:
                self.text_encoder = self.text_encoder.to(torch.device("cpu"))

            if not init_cache and self.current_start > 0:
                self._recache_frames()

        if not init_cache:
            return

        self.current_start = 0

        # Only support local attention
        kv_cache_size = self.local_attn_size * self.frame_seq_length

        generator_param = next(self.generator.model.parameters())
        self._initialize_kv_cache(
            batch_size=self.batch_size,
            dtype=generator_param.dtype,
            device=generator_param.device,
            kv_cache_size_override=kv_cache_size,
        )
        self._initialize_crossattn_cache(
            batch_size=self.batch_size,
            dtype=generator_param.dtype,
            device=generator_param.device,
        )

        self.generator.model.local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        self.vae.clear_cache()

        # Initialize recache buffer
        latent_height = self.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = self.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        self.recache_buffer = torch.zeros(
            [
                self.batch_size,
                self.local_attn_size,
                16,
                latent_height,
                latent_width,
            ],
            dtype=generator_param.dtype,
            device=generator_param.device
            if not self.low_memory
            else torch.device("cpu"),
        )

    @torch.no_grad()
    def __call__(
        self, _: torch.Tensor | list[torch.Tensor] | None = None
    ) -> torch.Tensor:
        # Ignore input

        latent_height = self.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = self.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        generator_param = next(self.generator.model.parameters())

        # Create generator from seed for reproducible generation
        rng = torch.Generator(device=generator_param.device).manual_seed(self.seed)

        noise = torch.randn(
            [
                self.batch_size,
                self.num_frame_per_block,
                16,
                latent_height,
                latent_width,
            ],
            device=generator_param.device,
            dtype=generator_param.dtype,
            generator=rng,
        )

        for index, current_timestep in enumerate(self.denoising_step_list):
            timestep = (
                torch.ones(
                    [self.batch_size, self.num_frame_per_block],
                    device=noise.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self.current_start * self.frame_seq_length,
                )
                next_timestep = self.denoising_step_list[index + 1]
                # Create noise with same shape and properties as denoised_pred
                flattened_pred = denoised_pred.flatten(0, 1)
                random_noise = torch.randn(
                    flattened_pred.shape,
                    device=flattened_pred.device,
                    dtype=flattened_pred.dtype,
                    generator=rng,
                )
                noise = self.scheduler.add_noise(
                    flattened_pred,
                    random_noise,
                    next_timestep
                    * torch.ones(
                        [self.batch_size * self.num_frame_per_block],
                        device=noise.device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noise,
                    conditional_dict=self.conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self.current_start * self.frame_seq_length,
                )

        # rerun with clean context to update cache
        context_timestep = torch.ones_like(timestep) * 0
        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=self.current_start * self.frame_seq_length,
        )

        # Push the generated latents to the recache buffer (sliding window)
        # Shift buffer left, append new frames at end
        self.recache_buffer = torch.cat(
            [
                self.recache_buffer[:, self.num_frame_per_block :],
                denoised_pred.to(self.recache_buffer.device),
            ],
            dim=1,
        )

        self.current_start += self.num_frame_per_block

        output = self.vae.decode_to_pixel(denoised_pred, use_cache=True)
        return postprocess_chunk(output)

    def _initialize_kv_cache(
        self, batch_size, dtype, device, kv_cache_size_override: int | None = None
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device
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
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        """
        target_size = int(local_attn_size_value) * self.frame_seq_length

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            self.generator.model.max_attention_size = target_size
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                module.max_attention_size = target_size
                updated_modules.append(name if name else module.__class__.__name__)

    def _recache_frames(self):
        # Reset kv cache
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            cache["k"].zero_()
            cache["v"].zero_()

        # Reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # Get the number of frames to recache (min of what we've generated and buffer size)
        num_recache_frames = min(self.current_start, self.local_attn_size)
        recache_start = self.current_start - num_recache_frames

        # With sliding window, most recent frames are always at the end
        generator_device = next(self.generator.model.parameters()).device
        recache_frames = (
            self.recache_buffer[:, -num_recache_frames:]
            .contiguous()
            .to(generator_device)
        )

        # Prepare blockwise causal mask
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=recache_frames.device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        self.generator.model.block_mask = block_mask

        context_timestep = (
            torch.ones(
                [self.batch_size, num_recache_frames],
                device=recache_frames.device,
                dtype=torch.int64,
            )
            * 0
        )
        self.generator(
            noisy_image_or_video=recache_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=recache_start * self.frame_seq_length,
        )

        # Reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False
