# Adapted from https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/video_encoders/hf/tae.py
"""Tiny AutoEncoder (TAE) wrapper for Wan2.1 models.

TAE is a lightweight alternative VAE architecture from the LightX2V project.
Unlike WanVAE, TAE is a completely different architecture - a much smaller/faster
model designed for quick encoding/decoding previews.

Key differences from WanVAE:
- Uses MemBlock for temporal memory (different from CausalConv3d caching)
- Has TPool/TGrow blocks for temporal downsampling/upsampling
- Much simpler architecture with 64 channels throughout encoder
- Approximately 4x temporal upscaling in decoder (TGrow blocks expand frames)

Streaming mode:
- TAE supports streaming decode via parallel processing with persistent MemBlock memory
- Each batch is processed in parallel (fast) while memory state is maintained across batches
- This provides both speed AND temporal continuity for smooth streaming
- First decode call has fewer output frames due to TGrow expansion and frame trimming (3 frames)
"""

import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD


@dataclass
class TAEEncoderCache:
    """Explicit encoder cache for TAE streaming.

    This cache holds the MemBlock memory state for streaming encoding.
    Create separate cache instances for independent encoding streams
    (e.g., VACE inactive vs reactive) to prevent memory pollution.

    Usage:
        cache = vae.create_encoder_cache()
        latent = vae.encode_to_latent(pixels, encoder_cache=cache)
        # Cache is updated in-place, reuse for next chunk in same stream
    """

    memory: list[torch.Tensor | None] | None = field(default=None)
    initialized: bool = field(default=False)


# Default checkpoint filenames for Wan 2.1 TAE
DEFAULT_TAE_FILENAME = "taew2_1.pth"
LIGHTTAE_FILENAME = "lighttaew2_1.pth"


def _conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    """Create a 3x3 Conv2d with padding."""
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    """Clamp activation using scaled tanh."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """Memory block that combines current input with past state."""

    def __init__(self, n_in: int, n_out: int, act_func: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            act_func,
            _conv(n_out, n_out),
            act_func,
            _conv(n_out, n_out),
        )
        self.skip = (
            nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        )
        self.act = act_func

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TPool(nn.Module):
    """Temporal pooling block that combines multiple frames."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class _TGrow(nn.Module):
    """Temporal growth block that expands to multiple frames."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def _apply_model_parallel_streaming(
    model: nn.Sequential,
    x: torch.Tensor,
    N: int,
    initial_mem: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    """Apply model in parallel mode with streaming memory support.

    This processes all frames in parallel (fast) while maintaining temporal
    continuity across batches by using initial memory from the previous batch.

    Args:
        model: nn.Sequential of blocks to apply
        x: input data reshaped to (N*T, C, H, W)
        N: batch size (for reshaping)
        initial_mem: Initial memory values for each MemBlock (from previous batch).
                    If None, uses zeros for first batch.

    Returns:
        Tuple of (NTCHW output tensor, list of final memory values for next batch)
    """
    # Count MemBlocks for memory initialization
    num_memblocks = sum(1 for b in model if isinstance(b, _MemBlock))

    # Initialize memory list if not provided
    if initial_mem is None:
        initial_mem = [None] * num_memblocks

    # Track which MemBlock we're at
    mem_idx = 0
    final_mem = []

    for b in model:
        if isinstance(b, _MemBlock):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)

            # Create memory: pad with initial_mem at t=0, then shift frames
            if initial_mem[mem_idx] is not None:
                # Use previous batch's last frame as initial memory
                init_frame = initial_mem[mem_idx].reshape(N, 1, C, H, W)
                mem = torch.cat([init_frame, _x[:, :-1]], dim=1).reshape(x.shape)
            else:
                # First batch - use zeros
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(
                    x.shape
                )

            # Save last frame for next batch (input before processing)
            final_mem.append(_x[:, -1:].reshape(N, C, H, W).clone())
            mem_idx += 1

            x = b(x, mem)
        else:
            x = b(x)

    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W), final_mem


def _apply_model_with_memblocks(
    model: nn.Sequential,
    x: torch.Tensor,
    parallel: bool = True,
    show_progress_bar: bool = False,
) -> torch.Tensor:
    """Apply a sequential model with memblocks to the given input (batch mode).

    Args:
        model: nn.Sequential of blocks to apply
        x: input data, of dimensions NTCHW
        parallel: unused, kept for API compatibility (always uses parallel)
        show_progress_bar: unused, kept for API compatibility

    Returns:
        NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"_apply_model_with_memblocks: TAE expects NTCHW, got {x.ndim}D"
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)
    result, _ = _apply_model_parallel_streaming(model, x, N, initial_mem=None)
    return result


class _TAEModel(nn.Module):
    """Tiny AutoEncoder model for Wan 2.1.

    This is a lightweight VAE designed for quick previews. It uses a different
    architecture than the standard WanVAE, with MemBlocks for temporal processing.

    Supports two decode modes:
    - Batch mode (decode_video): Process all frames at once
    - Streaming mode (stream_decode): Process frames incrementally with persistent memory
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
    ):
        """Initialize TAE model.

        Args:
            checkpoint_path: Path to weight file (.pth or .safetensors)
            decoder_time_upscale: Whether temporal upsampling is enabled for each block
            decoder_space_upscale: Whether spatial upsampling is enabled for each block
            patch_size: Input/output pixelshuffle patch-size (1 for Wan 2.1)
            latent_channels: Number of latent channels (16 for Wan 2.1)
        """
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3

        # Wan 2.1 uses ReLU activation
        act_func = nn.ReLU(inplace=True)

        # Encoder: 64 channels throughout, simple architecture
        self.encoder = nn.Sequential(
            _conv(self.image_channels * self.patch_size**2, 64),
            act_func,
            _TPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _TPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _TPool(64, 1),
            _conv(64, 64, stride=2, bias=False),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _MemBlock(64, 64, act_func),
            _conv(64, self.latent_channels),
        )

        # Decoder with configurable upscaling
        n_f = [256, 128, 64, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1
        self._decoder_time_upscale = decoder_time_upscale

        self.decoder = nn.Sequential(
            _Clamp(),
            _conv(self.latent_channels, n_f[0]),
            act_func,
            _MemBlock(n_f[0], n_f[0], act_func),
            _MemBlock(n_f[0], n_f[0], act_func),
            _MemBlock(n_f[0], n_f[0], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            _TGrow(n_f[0], 1),
            _conv(n_f[0], n_f[1], bias=False),
            _MemBlock(n_f[1], n_f[1], act_func),
            _MemBlock(n_f[1], n_f[1], act_func),
            _MemBlock(n_f[1], n_f[1], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            _TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            _conv(n_f[1], n_f[2], bias=False),
            _MemBlock(n_f[2], n_f[2], act_func),
            _MemBlock(n_f[2], n_f[2], act_func),
            _MemBlock(n_f[2], n_f[2], act_func),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            _TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            _conv(n_f[2], n_f[3], bias=False),
            act_func,
            _conv(n_f[3], self.image_channels * self.patch_size**2),
        )

        # Streaming state for parallel streaming encode/decode
        self._encoder_mem: list[torch.Tensor | None] | None = None
        self._decoder_mem: list[torch.Tensor | None] | None = None
        self._frames_output: int = 0  # Track output frames for trim handling

        if checkpoint_path is not None:
            ext = os.path.splitext(checkpoint_path)[1].lower()
            if ext == ".pth":
                state_dict = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
            elif ext == ".safetensors":
                state_dict = load_file(checkpoint_path, device="cpu")
            else:
                raise ValueError(
                    f"_TAEModel.__init__: Unsupported checkpoint format: {ext}. "
                    "Supported: .pth, .safetensors"
                )
            self.load_state_dict(self._patch_tgrow_layers(state_dict))

    def _patch_tgrow_layers(self, sd: dict) -> dict:
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch

        Returns:
            Patched state dict
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, _TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # Take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def clear_decode_state(self):
        """Clear decoder streaming state for a new sequence."""
        self._decoder_mem = None
        self._frames_output = 0

    def clear_encode_state(self):
        """Clear encoder streaming state for a new sequence."""
        self._encoder_mem = None

    def stream_encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode frames in streaming mode with persistent memory.

        This uses parallel processing within each batch for speed, while maintaining
        MemBlock memory across batches for smooth temporal continuity at chunk
        boundaries.

        Unlike encode_video, this maintains state across calls.
        Call clear_encode_state() before a new sequence.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1]

        Returns:
            NTCHW latent tensor with approximately Gaussian values
        """
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            # Pad at end to multiple of 4
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)

        N, T, C, H, W = x.shape
        x_flat = x.reshape(N * T, C, H, W)

        result, self._encoder_mem = _apply_model_parallel_streaming(
            self.encoder,
            x_flat,
            N,
            initial_mem=self._encoder_mem,
        )

        return result

    def encode_video(
        self,
        x: torch.Tensor,
        parallel: bool = True,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1]
            parallel: if True, all frames processed at once (faster, more memory)
                     if False, frames processed sequentially (slower, O(1) memory)
            show_progress_bar: if True, display tqdm progress bar

        Returns:
            NTCHW latent tensor with approximately Gaussian values
        """
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            # Pad at end to multiple of 4
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return _apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def decode_video(
        self,
        x: torch.Tensor,
        parallel: bool = True,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        """Decode a sequence of frames (batch mode).

        Args:
            x: input NTCHW latent tensor with approximately Gaussian values
            parallel: if True, all frames processed at once (faster, more memory)
                     if False, frames processed sequentially (slower, O(1) memory)
            show_progress_bar: if True, display tqdm progress bar

        Returns:
            NTCHW RGB tensor with values clamped to [0, 1]
        """
        x = _apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x[:, self.frames_to_trim :]

    def stream_decode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Decode frames in streaming mode with persistent memory.

        This uses parallel processing within each batch for speed, while maintaining
        MemBlock memory across batches for smooth temporal continuity.

        On the first batch, frames are processed sequentially (first frame separately,
        then remaining frames) to match WanVAE warmup behavior for better temporal
        consistency.

        Unlike decode_video, this maintains state across calls.
        Call clear_decode_state() before a new sequence.

        Args:
            x: input NTCHW latent tensor (typically 1-4 frames at a time)

        Returns:
            NTCHW RGB tensor with values in [0, 1].
            First call returns fewer frames due to temporal trim.
        """
        N, T, C, H, W = x.shape

        # First batch warmup: process first frame separately, then remaining frames
        # This matches WanVAE's warmup behavior for better temporal consistency
        if self._frames_output == 0:
            # Clear decoder memory state for first batch
            self._decoder_mem = None

            # Process first frame separately
            first_frame = x[:, :1, :, :, :]  # [N, 1, C, H, W]
            first_flat = first_frame.reshape(N * 1, C, H, W)
            first_result, first_mem = _apply_model_parallel_streaming(
                self.decoder,
                first_flat,
                N,
                initial_mem=None,  # Use zeros for first frame
            )

            # Process remaining frames if any
            if T > 1:
                remaining_frames = x[:, 1:, :, :, :]  # [N, T-1, C, H, W]
                remaining_flat = remaining_frames.reshape(N * (T - 1), C, H, W)
                remaining_result, self._decoder_mem = _apply_model_parallel_streaming(
                    self.decoder,
                    remaining_flat,
                    N,
                    initial_mem=first_mem,  # Use memory from first frame
                )
                # Concatenate first frame and remaining frames
                result = torch.cat([first_result, remaining_result], dim=1)
            else:
                # Only one frame
                result = first_result
                self._decoder_mem = first_mem
        else:
            # Subsequent batches: use parallel processing with persistent memory
            x_flat = x.reshape(N * T, C, H, W)
            result, self._decoder_mem = _apply_model_parallel_streaming(
                self.decoder,
                x_flat,
                N,
                initial_mem=self._decoder_mem,
            )

        result = result.clamp_(0, 1)

        if self.patch_size > 1:
            result = F.pixel_shuffle(result, self.patch_size)

        # Handle temporal trim - only trim on first output
        if self._frames_output == 0 and result.shape[1] > self.frames_to_trim:
            result = result[:, self.frames_to_trim :]

        self._frames_output += result.shape[1]

        return result


class TAEWrapper(nn.Module):
    """TAE wrapper with interface matching WanVAEWrapper.

    This provides a consistent interface for the Tiny AutoEncoder that matches
    the WanVAEWrapper's encode_to_latent/decode_to_pixel/clear_cache API.

    The wrapper can instantiate either the full TAE or LightTAE through the
    `use_lighttae` parameter. LightTAE uses WanVAE normalization constants for
    consistent latent space with WanVAE, while regular TAE has its own latent space.

    Streaming mode (use_cache=True):
        TAE maintains persistent MemBlock memory for smooth frame-by-frame streaming.
        This is faster than batch mode for real-time applications since it processes
        smaller chunks while maintaining temporal continuity.

    Batch mode (use_cache=False):
        Processes all frames at once without persistent state. Good for one-shot
        encoding/decoding of complete videos.

    Args:
        model_dir: Base directory containing model files
        model_name: Model subdirectory name (defaults to "Autoencoders")
        vae_path: Explicit path to TAE checkpoint (overrides model_dir/model_name)
        use_lighttae: If True, use LightTAE with WanVAE normalization (faster, lower quality)
    """

    def __init__(
        self,
        model_dir: str = "wan_models",
        model_name: str = "Autoencoders",
        vae_path: str | None = None,
        use_lighttae: bool = False,
    ):
        super().__init__()

        # Determine checkpoint path with priority: explicit vae_path > model_dir/model_name default
        # Both TAE and LightTAE downloaded from lightx2v/Autoencoders
        if vae_path is None:
            default_filename = (
                LIGHTTAE_FILENAME if use_lighttae else DEFAULT_TAE_FILENAME
            )
            vae_path = os.path.join(model_dir, model_name, default_filename)

        self.z_dim = 16
        self.use_lighttae = use_lighttae

        # Register normalization buffers for LightTAE (same as WanVAEWrapper)
        if use_lighttae:
            self.register_buffer(
                "mean", torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
            )
            self.register_buffer(
                "std", torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)
            )

        # Create TAE model
        self.model = (
            _TAEModel(
                checkpoint_path=vae_path,
                patch_size=1,
                latent_channels=self.z_dim,
            )
            .eval()
            .requires_grad_(False)
        )

        # Default encoder cache for backwards compatibility (when no explicit cache passed)
        self._default_encoder_cache = TAEEncoderCache()
        self._first_decode = True

    def create_encoder_cache(self) -> TAEEncoderCache:
        """Create a fresh encoder cache for streaming.

        Use separate caches for independent encoding streams to prevent
        TAE's MemBlock memory pollution. This is essential for VACE which
        encodes inactive and reactive portions separately.

        Returns:
            A new TAEEncoderCache instance for use with encode_to_latent()

        Example:
            # For VACE dual-encode
            inactive_cache = vae.create_encoder_cache()
            reactive_cache = vae.create_encoder_cache()
            inactive_latent = vae.encode_to_latent(inactive, encoder_cache=inactive_cache)
            reactive_latent = vae.encode_to_latent(reactive, encoder_cache=reactive_cache)
        """
        return TAEEncoderCache()

    def encode_to_latent(
        self,
        pixel: torch.Tensor,
        use_cache: bool = True,
        encoder_cache: TAEEncoderCache | None = None,
    ) -> torch.Tensor:
        """Encode video pixels to latents.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]
            use_cache: If True, use streaming encode with persistent memory.
                      If False, use batch encode (no persistent state).
            encoder_cache: Explicit cache for streaming mode. If None, uses internal
                          default cache. Pass separate cache instances for independent
                          encoding streams (e.g., VACE inactive vs reactive) to prevent
                          TAE's MemBlock memory pollution.

        Returns:
            Latent tensor [batch, frames, channels, height, width]

        Note:
            TAE's MemBlock architecture mixes processed memory state with input via
            channel concatenation. When encoding multiple interleaved streams (like
            VACE's inactive + reactive), each stream needs its own cache to maintain
            temporal continuity without cross-contamination.

            For LightTAE, normalization is applied to match WanVAE's latent space.
        """
        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w] for TAE
        pixel_ntchw = pixel.permute(0, 2, 1, 3, 4)

        # Scale from [-1, 1] to [0, 1] range expected by TAE
        pixel_ntchw = (pixel_ntchw + 1) / 2

        if use_cache:
            # Use provided cache or fall back to default
            cache = (
                encoder_cache
                if encoder_cache is not None
                else self._default_encoder_cache
            )

            # Initialize cache on first use
            if not cache.initialized:
                cache.memory = None
                cache.initialized = True

            # Restore cache memory to model
            self.model._encoder_mem = cache.memory

            latent = self.model.stream_encode(pixel_ntchw)

            # Save updated memory back to cache
            cache.memory = self.model._encoder_mem
        else:
            # Batch mode - no persistent state
            latent = self.model.encode_video(
                pixel_ntchw, parallel=True, show_progress_bar=False
            )

        # Apply normalization for LightTAE to match WanVAE latent space
        # This follows LightX2V's approach: normalized = (latent - mean) / std
        if self.use_lighttae:
            # Convert to [batch, channels, frames, h, w] format for normalization
            latent = latent.permute(0, 2, 1, 3, 4)

            device, dtype = latent.device, latent.dtype
            mean = self.mean.to(device=device, dtype=dtype)
            std = self.std.to(device=device, dtype=dtype)

            # Normalize: (latent - mean) / std
            # Latent is [batch, channels, frames, h, w], so view as (1, z_dim, 1, 1, 1)
            latent = (latent - mean.view(1, self.z_dim, 1, 1, 1)) / std.view(
                1, self.z_dim, 1, 1, 1
            )

            # Convert back to [batch, frames, channels, h, w] format for return
            latent = latent.permute(0, 2, 1, 3, 4)

        # Return in [batch, frames, channels, h, w] format
        return latent

    def _get_scale(self, device: torch.device, dtype: torch.dtype) -> list:
        """Get normalization scale parameters on the correct device/dtype.

        Returns [mean, std] for denormalization: latent * std + mean
        Note: This differs from WanVAEWrapper which returns [mean, 1/std] for its
        internal decode method. LightTAE denormalization matches LightX2V's formula.
        """
        return [
            self.mean.to(device=device, dtype=dtype),
            self.std.to(device=device, dtype=dtype),
        ]

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        """Decode latents to video pixels.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width]
            use_cache: If True, use streaming decode with persistent memory.
                      If False, use batch decode (clears state).

        Returns:
            Video tensor [batch, frames, channels, height, width] in range [-1, 1]

        Note:
            In streaming mode (use_cache=True), TAE maintains MemBlock state across
            calls for smooth temporal continuity. Uses parallel processing within
            each batch for speed. The first call may have fewer output frames due
            to TGrow temporal expansion and frame trimming.

            For LightTAE (use_lighttae=True), denormalization is applied during decode
            to convert from the normalized latent space back to WanVAE's latent space
            distribution, following LightX2V's formula: latent * std + mean
        """
        # Apply denormalization for LightTAE (reverse of encoding normalization)
        if self.use_lighttae:
            # [batch, frames, channels, h, w] -> [batch, channels, frames, h, w] (match LightX2V format)
            latent = latent.permute(0, 2, 1, 3, 4)

            device, dtype = latent.device, latent.dtype
            mean = self.mean.to(device=device, dtype=dtype)
            std = self.std.to(device=device, dtype=dtype)

            # Denormalize: latent * std + mean (inverse of encoding normalization)
            # This matches LightX2V's formula: latents / (1/std) + mean = latents * std + mean
            # Latent is [batch, channels, frames, h, w], so view as (1, z_dim, 1, 1, 1)
            latent = latent * std.view(1, self.z_dim, 1, 1, 1) + mean.view(
                1, self.z_dim, 1, 1, 1
            )

            # Convert back to [batch, frames, channels, h, w] for TAE decode
            latent = latent.permute(0, 2, 1, 3, 4)

        if use_cache:
            # Streaming mode - use parallel processing with persistent memory
            if self._first_decode:
                self.model.clear_decode_state()
                self._first_decode = False

            output = self.model.stream_decode(latent)
        else:
            # Batch mode - no persistent state
            output = self.model.decode_video(
                latent, parallel=True, show_progress_bar=False
            )

        # Scale from [0, 1] to [-1, 1] range
        output = output * 2 - 1
        output = output.clamp_(-1, 1)

        # Return in [batch, frames, channels, h, w] format
        return output

    def clear_cache(self):
        """Clear state for next sequence."""
        # Reset default encoder cache
        self._default_encoder_cache = TAEEncoderCache()
        self._first_decode = True
        self.model.clear_encode_state()
        self.model.clear_decode_state()
