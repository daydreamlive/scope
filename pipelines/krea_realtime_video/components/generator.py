# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from einops import rearrange


class KreaGeneratorWrapper(torch.nn.Module):
    """
    KV Cache component for managing KV cache and cross-attention cache.
    Depends on the generator module to access model configuration.
    """

    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.generator = generator
        self.kv_cache1 = None
        self.crossattn_cache = None

        # Attributes moved from stream component for recompute functionality
        self.frame_seq_length = None
        self.kv_cache_num_frames = None
        self.num_frame_per_block = None
        self.batch_size = None
        self.conditional_dict = None
        self.current_start = 0
        self.context_frame_buffer = None
        self.context_frame_buffer_max_size = 0
        self.first_context_frame = None
        self.decoded_frame_buffer = None
        self.vae = None
        self.decoded_frame_buffer_max_size = None

    def configure(
        self,
        frame_seq_length: int,
        batch_size: int,
        num_frame_per_block: int,
        kv_cache_num_frames: int,
        vae: torch.nn.Module,
    ):
        """
        Configure the KV cache component with necessary parameters.

        Args:
            frame_seq_length: Frame sequence length
            batch_size: Batch size for the cache
            num_frame_per_block: Number of frames per block
            kv_cache_num_frames: Number of frames to cache
            vae: VAE module for encoding/decoding
        """
        self.frame_seq_length = frame_seq_length
        self.batch_size = batch_size
        self.num_frame_per_block = num_frame_per_block
        self.kv_cache_num_frames = kv_cache_num_frames
        self.vae = vae
        self.current_start = 0
        self.conditional_dict = None
        self.context_frame_buffer = None
        self.context_frame_buffer_max_size = kv_cache_num_frames - 1
        self.first_context_frame = None
        self.decoded_frame_buffer = None
        self.decoded_frame_buffer_max_size = 1 + (kv_cache_num_frames - 1) * 4

    @property
    def model(self):
        """Delegate model access to the wrapped generator."""
        return self.generator.model

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        classify_mode: bool | None = False,
        concat_time_embeddings: bool | None = False,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        cache_start: int | None = None,
        kv_cache_attention_bias: float = 1.0,
    ):
        """
        Forward pass that delegates to the wrapped generator.
        """
        return self.generator(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            classify_mode=classify_mode,
            concat_time_embeddings=concat_time_embeddings,
            clean_x=clean_x,
            aug_t=aug_t,
            cache_start=cache_start,
            kv_cache_attention_bias=kv_cache_attention_bias,
        )

    @property
    def num_transformer_blocks(self) -> int:
        """Get the number of transformer blocks from the generator model."""
        return len(self.generator.model.blocks)

    @torch.no_grad()
    def initialize_kv_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_cache_size_override: int | None = None,
        local_attn_size: int | None = None,
        frame_seq_length: int | None = None,
    ):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Args:
            batch_size: Batch size for the cache
            dtype: Data type for the cache tensors
            device: Device to create the cache on
            kv_cache_size_override: Optional override for cache size
            local_attn_size: Local attention size (used if kv_cache_size_override is None)
            frame_seq_length: Frame sequence length (used if kv_cache_size_override is None)
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if local_attn_size is not None and local_attn_size != -1:
                # Local attention: cache only needs to store the window
                if frame_seq_length is None:
                    raise ValueError(
                        "frame_seq_length must be provided when local_attn_size is not -1"
                    )
                kv_cache_size = local_attn_size * frame_seq_length
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

    @torch.no_grad()
    def initialize_crossattn_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.

        Args:
            batch_size: Batch size for the cache
            dtype: Data type for the cache tensors
            device: Device to create the cache on
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

    def _get_context_frames(self) -> torch.Tensor:
        """
        Get context frames for cache recomputation.
        Returns the appropriate context frames based on current_start position.
        """
        if self.first_context_frame is None:
            raise ValueError("first_context_frame must be set before recomputing cache")
        if self.kv_cache_num_frames is None:
            raise ValueError("kv_cache_num_frames must be set before recomputing cache")
        if self.num_frame_per_block is None:
            raise ValueError("num_frame_per_block must be set before recomputing cache")

        generator_device = next(self.generator.model.parameters()).device
        if (self.current_start - self.num_frame_per_block) < self.kv_cache_num_frames:
            if self.kv_cache_num_frames == 1:
                # The context just contains the first frame
                return self.first_context_frame
            else:
                # The context contains first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
                if self.context_frame_buffer is None:
                    raise ValueError(
                        "context_frame_buffer must be set before recomputing cache"
                    )
                return torch.cat(
                    [
                        self.first_context_frame,
                        self.context_frame_buffer.to(generator_device),
                    ],
                    dim=1,
                )
        else:
            # The context contains the re-encoded first frame + the kv_cache_num_frames - 1 frames in the context frame buffer
            if self.vae is None or self.decoded_frame_buffer is None:
                raise ValueError(
                    "vae and decoded_frame_buffer must be set for cache recomputation "
                    "when current_start exceeds kv_cache_num_frames"
                )
            if self.context_frame_buffer is None:
                raise ValueError(
                    "context_frame_buffer must be set before recomputing cache"
                )
            vae_device = next(self.vae.parameters()).device
            decoded_first_frame = self.decoded_frame_buffer[:, :1].to(vae_device)
            reencoded_latent = self.vae.encode_to_latent(
                rearrange(decoded_first_frame, "B T C H W -> B C T H W")
            )
            return torch.cat(
                [reencoded_latent, self.context_frame_buffer.to(generator_device)],
                dim=1,
            )

    @torch.no_grad()
    def _recompute_cache(self):
        """
        Recompute KV cache using context frames.
        This method should be called when current_start > 0 (not the first frame).
        """
        if self.current_start == 0:
            return

        if self.kv_cache_num_frames is None:
            raise ValueError("kv_cache_num_frames must be set before recomputing cache")
        if self.frame_seq_length is None:
            raise ValueError("frame_seq_length must be set before recomputing cache")
        if self.batch_size is None:
            raise ValueError("batch_size must be set before recomputing cache")
        if self.conditional_dict is None:
            raise ValueError("conditional_dict must be set before recomputing cache")

        start_frame = min(self.current_start, self.kv_cache_num_frames)

        # With sliding window, most recent frames are always at the end
        context_frames = self._get_context_frames()
        num_context_frames = context_frames.shape[1]

        self.initialize_kv_cache(
            batch_size=self.batch_size,
            dtype=context_frames.dtype,
            device=context_frames.device,
            local_attn_size=-1,  # Use global attention for recompute
            frame_seq_length=self.frame_seq_length,
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
        # Cache recomputation: no bias to faithfully store context frames
        self.generator(
            noisy_image_or_video=context_frames,
            conditional_dict=self.conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            crossattn_cache=self.crossattn_cache,
            current_start=start_frame * self.frame_seq_length,
        )

        self.generator.model.block_mask = None

    def initialize_cache_buffers(
        self,
        height: int,
        width: int,
        low_memory: bool = False,
        vae_spatial_downsample_factor: int = 8,
    ):
        """
        Initialize context frame buffer and decoded frame buffer.

        Args:
            height: Image height
            width: Image width
            low_memory: Whether to use low memory mode
            vae_spatial_downsample_factor: VAE spatial downsampling factor
        """
        generator_param = next(self.generator.model.parameters())
        latent_height = height // vae_spatial_downsample_factor
        latent_width = width // vae_spatial_downsample_factor

        self.context_frame_buffer = torch.zeros(
            [
                self.batch_size,
                self.context_frame_buffer_max_size,
                16,
                latent_height,
                latent_width,
            ],
            dtype=generator_param.dtype,
            device=generator_param.device if not low_memory else torch.device("cpu"),
        )

        self.decoded_frame_buffer = torch.zeros(
            [
                self.batch_size,
                self.decoded_frame_buffer_max_size,
                3,
                height,
                width,
            ],
            dtype=generator_param.dtype,
            device=generator_param.device if not low_memory else torch.device("cpu"),
        )

    def initialize_full_cache(
        self,
        local_attn_size: int,
        height: int,
        width: int,
        low_memory: bool = False,
        max_rope_freq_table_seq_len: int = 1024,
        vae_spatial_downsample_factor: int = 8,
    ):
        """
        Initialize the full KV cache system including buffers and model configuration.

        Args:
            local_attn_size: Local attention size
            height: Image height
            width: Image width
            low_memory: Whether to use low memory mode
            max_rope_freq_table_seq_len: Maximum RoPE frequency table sequence length
            vae_spatial_downsample_factor: VAE spatial downsampling factor
        """
        generator_param = next(self.generator.model.parameters())

        # CausalWanModel uses a RoPE frequency table with a max sequence length
        # We need to make sure that current_start does not shift past the max sequence length
        max_current_start = max_rope_freq_table_seq_len - self.num_frame_per_block
        if self.current_start >= max_current_start:
            self.current_start = 0

        self.current_start = 0
        self.first_context_frame = None

        for block in self.generator.model.blocks:
            block.self_attn.local_attn_size = -1
            block.self_attn.num_frame_per_block = self.num_frame_per_block

        self.generator.model.local_attn_size = local_attn_size

        self._set_all_modules_frame_seq_length(self.frame_seq_length)
        self._set_all_modules_max_attention_size(local_attn_size)

        kv_cache_size = local_attn_size * self.frame_seq_length

        self.initialize_kv_cache(
            batch_size=self.batch_size,
            dtype=generator_param.dtype,
            device=generator_param.device,
            kv_cache_size_override=kv_cache_size,
            local_attn_size=local_attn_size,
            frame_seq_length=self.frame_seq_length,
        )
        self.initialize_crossattn_cache(
            batch_size=self.batch_size,
            dtype=generator_param.dtype,
            device=generator_param.device,
        )

        if self.vae is not None:
            self.vae.clear_cache()

        self.initialize_cache_buffers(
            height=height,
            width=width,
            low_memory=low_memory,
            vae_spatial_downsample_factor=vae_spatial_downsample_factor,
        )

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        """
        target_size = int(local_attn_size_value) * self.frame_seq_length

        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            self.generator.model.max_attention_size = target_size

        # Update all child modules
        for _name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                module.max_attention_size = target_size

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

    def delegate_attribute_access(self, name: str):
        """
        Delegate attribute access to the underlying generator.
        """
        return getattr(self.generator, name)
