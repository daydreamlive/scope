"""Unified Wan VAE wrapper with streaming and batch encoding/decoding.

This module provides a unified VAE wrapper that supports both the full WanVAE
and the 75% pruned LightVAE through a single interface with a `use_lightvae` parameter.
"""

import logging
import os

import torch

from .constants import WAN_VAE_LATENT_MEAN, WAN_VAE_LATENT_STD
from .modules.vae import _video_vae, count_conv3d

logger = logging.getLogger(__name__)

# Default filenames for VAE checkpoints
DEFAULT_VAE_FILENAME = "Wan2.1_VAE.pth"
LIGHTVAE_FILENAME = "lightvaew2_1.pth"

# LightVAE pruning rate (75% of channels pruned)
LIGHTVAE_PRUNING_RATE = 0.75


class WanVAEWrapper(torch.nn.Module):
    """Unified VAE wrapper for Wan2.1 models.

    This VAE supports both streaming (cached) and batch encoding/decoding modes.
    Normalization is always applied during encoding for consistent latent distributions.

    The wrapper can instantiate either the full WanVAE or the 75% pruned LightVAE
    through the `use_lightvae` parameter.

    Args:
        model_dir: Base directory containing model files
        model_name: Model subdirectory name (e.g., "Wan2.1-T2V-1.3B")
        vae_path: Explicit path to VAE checkpoint (overrides model_dir/model_name)
        use_lightvae: If True, use 75% pruned LightVAE (faster, lower quality)
    """

    def __init__(
        self,
        model_dir: str = "wan_models",
        model_name: str = "Wan2.1-T2V-1.3B",
        vae_path: str | None = None,
        use_lightvae: bool = False,
    ):
        super().__init__()

        # Determine pruning rate based on VAE type
        pruning_rate = LIGHTVAE_PRUNING_RATE if use_lightvae else 0.0

        # Determine paths with priority: explicit vae_path > model_dir/model_name default
        if vae_path is None:
            filename = LIGHTVAE_FILENAME if use_lightvae else DEFAULT_VAE_FILENAME
            vae_path = os.path.join(model_dir, model_name, filename)

        self.register_buffer(
            "mean", torch.tensor(WAN_VAE_LATENT_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "std", torch.tensor(WAN_VAE_LATENT_STD, dtype=torch.float32)
        )
        self.z_dim = 16

        self.model = (
            _video_vae(
                pretrained_path=vae_path,
                z_dim=self.z_dim,
                pruning_rate=pruning_rate,
            )
            .eval()
            .requires_grad_(False)
        )

        # Cache encoder conv count for dynamic cache sizing
        self._encoder_conv_count = count_conv3d(self.model.encoder)

    def _get_scale(self, device: torch.device, dtype: torch.dtype) -> list:
        """Get normalization scale parameters on the correct device/dtype."""
        return [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

    def _apply_encoding_normalization(
        self, latent: torch.Tensor, scale: list
    ) -> torch.Tensor:
        """Apply normalization to encoded latents."""
        if isinstance(scale[0], torch.Tensor):
            return (latent - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1
            )
        return (latent - scale[0]) * scale[1]

    def _create_encoder_cache(self) -> list:
        """Create a fresh encoder feature cache with dynamic sizing."""
        return [None] * self._encoder_conv_count

    def create_encoder_cache(self):
        """Create encoder cache (TAE compatibility).

        WanVAE's CausalConv3d uses raw frame prepending for temporal context,
        so it doesn't have TAE's MemBlock memory pollution issue. This method
        exists for interface compatibility with TAEWrapper.

        Returns:
            None (WanVAE doesn't need explicit caches)
        """
        return None

    def encode_to_latent(
        self,
        pixel: torch.Tensor,
        use_cache: bool = True,
        encoder_cache=None,
    ) -> torch.Tensor:
        """Encode video pixels to latents.

        Args:
            pixel: Input video tensor [batch, channels, frames, height, width]
            use_cache: If True, use streaming encode (maintains cache state).
                      If False, use batch encode with a temporary cache.
            encoder_cache: Ignored (TAE compatibility). WanVAE's CausalConv3d
                          prepends raw frames as context, avoiding the MemBlock
                          memory pollution issue that requires separate caches.

        Returns:
            Latent tensor [batch, frames, channels, height, width]
        """
        device, dtype = pixel.device, pixel.dtype
        scale = self._get_scale(device, dtype)

        if use_cache:
            # Streaming encode - cache is maintained across calls
            latent = self.model.stream_encode(pixel)
            # Apply normalization (stream_encode returns unnormalized)
            latent = self._apply_encoding_normalization(latent, scale)
        else:
            # Batch encode with one-time cache (does not affect streaming state)
            # Create a temporary cache for the one-time encode
            latent = self._encode_with_cache(pixel, scale, self._create_encoder_cache())

        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return latent.permute(0, 2, 1, 3, 4)

    def _encode_with_cache(
        self, x: torch.Tensor, scale: list, feat_cache: list
    ) -> torch.Tensor:
        """Encode using a temporary cache without affecting internal streaming state.

        Always uses the eager feat_cache path on the uncompiled encoder.
        This is a one-off operation (reference images, etc.) where compilation
        overhead would be wasted, and the variable frame shapes would cause
        recompilation of the steady-state-only compiled graph.
        """
        t = x.shape[2]

        # Bypass torch.compile to avoid recompilation from non-steady-state shapes
        enc = getattr(self.model.encoder, "_orig_mod", self.model.encoder)

        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            conv_idx = [0]
            if i == 0:
                out = enc(
                    x[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
            else:
                out_ = enc(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                    feat_idx=conv_idx,
                )
                out = torch.cat([out, out_], 2)

        mu, _ = self.model.conv1(out).chunk(2, dim=1)
        return self._apply_encoding_normalization(mu, scale)

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        """Decode latents to video pixels.

        Args:
            latent: Latent tensor [batch, frames, channels, height, width]
            use_cache: If True, use streaming decode (maintains cache state).
                      If False, use batch decode (clears cache before/after).

        Returns:
            Video tensor [batch, frames, channels, height, width] in range [-1, 1]
        """
        # [batch, frames, channels, h, w] -> [batch, channels, frames, h, w]
        zs = latent.permute(0, 2, 1, 3, 4)
        zs = zs.to(torch.bfloat16).to("cuda")

        device, dtype = latent.device, latent.dtype
        scale = self._get_scale(device, dtype)

        if use_cache:
            output = self.model.stream_decode(zs, scale)
        else:
            output = self.model.decode(zs, scale)

        output = output.float().clamp_(-1, 1)
        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return output.permute(0, 2, 1, 3, 4)

    def clear_cache(self):
        """Clear encoder/decoder cache for next sequence."""
        self.model.first_batch = True

    def compile_decoder(self, height: int, width: int, num_frames: int = 12):
        """Apply torch.compile to the decoder and conv2, then warmup.

        Compiles the steady-state streaming decode path (cache_bufs) for ~1.4x
        speedup. First-time compilation takes several minutes (triton kernel
        generation); subsequent runs use the cached kernels.

        Args:
            height: Output video height in pixels (needed for warmup latent shape).
            width: Output video width in pixels (needed for warmup latent shape).
            num_frames: Pixel frames per chunk (default 12). Used to derive
                the latent frame count for warmup shapes.
        """
        import time

        # Derive latent frame count: first chunk produces (num_frames/4) latent
        # frames after temporal downsampling (stride 2 twice).
        latent_frames = num_frames // 4
        latents = torch.zeros(
            1,
            latent_frames,
            self.z_dim,
            height // 8,
            width // 8,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Run first_batch pass EAGERLY (before compile) to allocate decoder
        # cache buffers. The cache path uses @torch.compiler.disable which
        # causes graph breaks; running it before compile prevents those breaks
        # from polluting dynamo's recompile counters.
        # Use no_grad to match generation context (avoids grad_mode guard failure).
        self.model.first_batch = True
        with torch.no_grad():
            self.decode_to_pixel(latents, use_cache=True)

        # Now compile. With cache buffers pre-allocated, stream_decode's
        # first_batch path will use cache_bufs (no graph breaks).
        logger.info("Compiling VAE decoder with torch.compile...")
        start = time.time()

        self.model.decoder = torch.compile(
            self.model.decoder, mode="max-autotune-no-cudagraphs", fullgraph=False
        )
        self.model.conv2 = torch.compile(
            self.model.conv2, mode="max-autotune-no-cudagraphs", fullgraph=False
        )

        # Warmup the compiled cache_bufs path under no_grad to match generation.
        with torch.no_grad():
            for _ in range(9):
                self.decode_to_pixel(latents, use_cache=True)

        # Reset for real generation
        self.model.first_batch = True

        logger.info(f"VAE decoder compilation completed in {time.time() - start:.2f}s")

    def compile_encoder(self, height: int, width: int, num_frames: int = 12):
        """Apply torch.compile to the encoder, then warmup.

        Compiles only the steady-state streaming encode path (4-frame chunks
        via cache_bufs). First batch always runs eagerly via _orig_mod bypass,
        so only the single steady-state shape (4 pixel frames) needs tracing.

        Args:
            height: Input video height in pixels (needed for warmup shape).
            width: Input video width in pixels (needed for warmup shape).
            num_frames: Pixel frames per chunk (default 12). Warmup traces
                the 4-frame steady-state shape.
        """
        import time

        pixels = torch.zeros(
            1, 3, num_frames, height, width, device="cuda", dtype=torch.bfloat16
        )

        # Eager first_batch pass to allocate encoder cache buffers.
        # stream_encode's first_batch always uses the eager cache= path
        # (bypassing torch.compile), so this just populates CacheState.
        # Use no_grad to match generation context (avoids grad_mode guard failure).
        self.model.first_batch = True
        with torch.no_grad():
            self.encode_to_latent(pixels, use_cache=True)

        logger.info("Compiling VAE encoder with torch.compile...")
        start = time.time()

        self.model.encoder = torch.compile(
            self.model.encoder, mode="max-autotune-no-cudagraphs", fullgraph=False
        )

        # Warmup only the steady-state path (4-frame chunks via cache_bufs).
        # first_batch doesn't auto-clear in stream_encode, so set it explicitly.
        # Use no_grad to match generation context (avoids grad_mode guard failure).
        self.model.first_batch = False
        with torch.no_grad():
            for _ in range(9):
                self.encode_to_latent(pixels, use_cache=True)

        # Reset for real generation
        self.model.first_batch = True

        logger.info(f"VAE encoder compilation completed in {time.time() - start:.2f}s")
