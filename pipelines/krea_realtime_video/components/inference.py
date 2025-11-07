# Contains code based on https://github.com/NVlabs/LongLive
import logging

import torch
from einops import rearrange

from ...process import postprocess_chunk

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
        self.base_seed = seed
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            config.denoising_step_list, dtype=torch.long
        )
        if config.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = len(self.generator.model.blocks)
        self.frame_seq_length = (self.height // SCALE_SIZE) * (self.width // SCALE_SIZE)

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.config = config
        self.batch_size = 1

        self.num_frame_per_block = config.get("num_frame_per_block", 1)
        self.kv_cache_num_frames = config.get("kv_cache_num_frames", 3)
        self.local_attn_size = self.kv_cache_num_frames + self.num_frame_per_block

        self.context_frame_buffer = None
        # Track the latest kv_cache_num_frames - 1 frames because we will
        # also concat this with the first frame during cache recomp
        self.context_frame_buffer_max_size = self.kv_cache_num_frames - 1
        self.decoded_frame_buffer = None
        self.decoded_frame_buffer_max_size = 1 + (self.kv_cache_num_frames - 1) * 4
        self.first_context_frame = None

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.conditional_dict = None
        self.current_start = 0

    @torch.no_grad()
    def prepare(
        self,
        prompts: list[str] = None,
        denoising_step_list: list[int] = None,
        init_cache: bool = False,
    ):
        generator_param = next(self.generator.model.parameters())

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
            self.text_encoder = self.text_encoder.to(generator_param.device)

            self.conditional_dict = self.text_encoder(text_prompts=prompts)
            if self.batch_size > 1:
                self.conditional_dict["prompt_embeds"] = self.conditional_dict[
                    "prompt_embeds"
                ].repeat(self.batch_size, 1, 1)

            # If in low memory mode offload text encoder to CPU
            if self.low_memory:
                self.text_encoder = self.text_encoder.to(torch.device("cpu"))

            if self.crossattn_cache is not None:
                self._initialize_crossattn_cache(
                    self.batch_size, generator_param.dtype, generator_param.device
                )

        if denoising_step_list is not None:
            self.denoising_step_list = torch.tensor(
                denoising_step_list, dtype=torch.long
            )

        if not init_cache:
            return

        self.current_start = 0
        self.first_context_frame = None

        for block in self.generator.model.blocks:
            block.self_attn.local_attn_size = -1
            block.self_attn.num_frame_per_block = self.num_frame_per_block

        self.generator.model.local_attn_size = self.local_attn_size

        self._set_all_modules_frame_seq_length(self.frame_seq_length)
        self._set_all_modules_max_attention_size(self.local_attn_size)

        kv_cache_size = self.local_attn_size * self.frame_seq_length

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

        self.vae.clear_cache()

        latent_height = self.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = self.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        self.context_frame_buffer = torch.zeros(
            [
                self.batch_size,
                self.context_frame_buffer_max_size,
                16,
                latent_height,
                latent_width,
            ],
            dtype=generator_param.dtype,
            device=generator_param.device
            if not self.low_memory
            else torch.device("cpu"),
        )

        self.decoded_frame_buffer = torch.zeros(
            [
                self.batch_size,
                self.decoded_frame_buffer_max_size,
                3,
                self.height,
                self.width,
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
        if self.current_start > 0:
            self._recompute_cache()

        latent_height = self.height // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        latent_width = self.width // VAE_SPATIAL_DOWNSAMPLE_FACTOR
        generator_param = next(self.generator.model.parameters())

        # Create generator from seed for reproducible generation
        # Derive unique seed per block of latents using current_start as offset
        frame_seed = self.base_seed + self.current_start
        rng = torch.Generator(device=generator_param.device).manual_seed(frame_seed)

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

        start_frame = min(self.current_start, self.kv_cache_num_frames)

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
                    current_start=start_frame * self.frame_seq_length,
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
                    current_start=start_frame * self.frame_seq_length,
                )

        if self.current_start == 0:
            self.first_context_frame = denoised_pred[:, :1]

        # Push the generated latents to the context frame buffer (sliding window)
        if self.context_frame_buffer_max_size > 0:
            self.context_frame_buffer = torch.cat(
                [
                    self.context_frame_buffer,
                    denoised_pred.to(
                        self.context_frame_buffer.device,
                        self.context_frame_buffer.dtype,
                    ),
                ],
                dim=1,
            )[:, -self.context_frame_buffer_max_size :]

        output = self.vae.decode_to_pixel(denoised_pred, use_cache=True)

        # Push the decoded frames to the decoded frame buffer (sliding window)
        self.decoded_frame_buffer = torch.cat(
            [
                self.decoded_frame_buffer,
                output.to(
                    self.decoded_frame_buffer.device, self.decoded_frame_buffer.dtype
                ),
            ],
            dim=1,
        )[:, -self.decoded_frame_buffer_max_size :]

        self.current_start += self.num_frame_per_block

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

        # Get num_heads and head_dim from the generator model
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads

        if self.kv_cache1:
            for i in range(self.num_transformer_blocks):
                self.kv_cache1[i]["k"].zero_()
                self.kv_cache1[i]["v"].zero_()
                self.kv_cache1[i]["global_end_index"] = 0
                self.kv_cache1[i]["local_end_index"] = 0
        else:
            for _ in range(self.num_transformer_blocks):
                kv_cache1.append(
                    {
                        "k": torch.zeros(
                            [batch_size, kv_cache_size, num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, kv_cache_size, num_heads, head_dim],
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

        # Get num_heads and head_dim from the generator model
        num_heads = self.generator.model.num_heads
        head_dim = self.generator.model.dim // num_heads

        if self.crossattn_cache:
            for i in range(self.num_transformer_blocks):
                self.crossattn_cache[i]["k"].zero_()
                self.crossattn_cache[i]["v"].zero_()
                self.crossattn_cache[i]["is_init"] = False
        else:
            for _ in range(self.num_transformer_blocks):
                crossattn_cache.append(
                    {
                        "k": torch.zeros(
                            [batch_size, 512, num_heads, head_dim],
                            dtype=dtype,
                            device=device,
                        ),
                        "v": torch.zeros(
                            [batch_size, 512, num_heads, head_dim],
                            dtype=dtype,
                            device=device,
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

    def _set_all_modules_frame_seq_length(self, frame_seq_length: int):
        """
        Set frame_seq_length on all submodules that define it.
        """
        if hasattr(self.generator, "seq_len") and hasattr(
            self.generator.model, "local_attn_size"
        ):
            local_attn_size = self.generator.model.local_attn_size
            if local_attn_size > 21:
                self.generator.seq_len = frame_seq_length * local_attn_size
            else:
                self.generator.seq_len = 32760

        # Update root model if applicable
        if hasattr(self.generator.model, "frame_seq_length"):
            self.generator.model.frame_seq_length = frame_seq_length

        # Update all child modules (especially CausalWanSelfAttention instances)
        for _, module in self.generator.model.named_modules():
            if hasattr(module, "frame_seq_length"):
                module.frame_seq_length = frame_seq_length

    def _get_context_frames(self) -> torch.Tensor:
        generator_device = next(self.generator.model.parameters()).device
        if (self.current_start - self.num_frame_per_block) < self.kv_cache_num_frames:
            if self.kv_cache_num_frames == 1:
                # The context just contains the first frame
                return self.first_context_frame
            else:
                # The context contains first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
                return torch.cat(
                    [
                        self.first_context_frame,
                        self.context_frame_buffer.to(generator_device),
                    ],
                    dim=1,
                )
        else:
            # The context contains the re-encoded first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
            vae_device = next(self.vae.parameters()).device
            decoded_first_frame = self.decoded_frame_buffer[:, :1].to(vae_device)
            reencoded_latent = self.vae.encode_to_latent(
                rearrange(decoded_first_frame, "B T C H W -> B C T H W")
            )
            return torch.cat(
                [reencoded_latent, self.context_frame_buffer.to(generator_device)],
                dim=1,
            )

    def _recompute_cache(self):
        start_frame = min(self.current_start, self.kv_cache_num_frames)

        # With sliding window, most recent frames are always at the end
        context_frames = self._get_context_frames()
        num_context_frames = context_frames.shape[1]

        self._initialize_kv_cache(
            self.batch_size, context_frames.dtype, context_frames.device
        )

        # Prepare blockwise causal mask
        self.generator.model.block_mask = (
            self.generator.model._prepare_blockwise_causal_attn_mask(
                device=context_frames.device,
                num_frames=num_context_frames,
                frame_seqlen=self.frame_seq_length,
                num_frame_per_block=self.num_frame_per_block,
                local_attn_size=-1,
            )
        )

        context_timestep = (
            torch.ones(
                [self.batch_size, num_context_frames],
                device=context_frames.device,
                dtype=torch.int64,
            )
            * 0
        )
        self.generator(
            noisy_image_or_video=context_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=start_frame * self.frame_seq_length,
        )

        self.generator.model.block_mask = None
