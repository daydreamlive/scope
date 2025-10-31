import torch


class StreamInnerModelAdapter:
    """
    Inner model adapter that provides .stream_encode() and cache management.

    If the base wrapper's inner model has native stream_encode/stream_decode methods,
    uses them directly for optimal streaming performance. Otherwise, falls back to
    adapter methods using encode_to_latent/decode_to_pixel.
    """

    def __init__(self, base_vae_wrapper):
        self.base_wrapper = base_vae_wrapper

        self._has_native_streaming = (
            hasattr(base_vae_wrapper, 'model')
            and hasattr(base_vae_wrapper.model, 'stream_encode')
            and hasattr(base_vae_wrapper.model, 'stream_decode')
        )

        if self._has_native_streaming:
            self.first_batch = base_vae_wrapper.model.first_batch
        else:
            self.first_batch = True

    def stream_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        stream_encode: Stream encode with cache management.

        Args:
            x: [B, C, T, H, W] pixel tensor

        Returns:
            [B, Cz, T, H/8, W/8] latent tensor (note channel/time order!)
        """
        if self._has_native_streaming:
            return self.base_wrapper.model.stream_encode(x)
        else:
            if self.first_batch:
                self.base_wrapper.clear_cache()
                self.first_batch = False

            latent = self.base_wrapper.encode_to_latent(x)
            return latent.permute(0, 2, 1, 3, 4)

    def clear_cache_encode(self):
        if self._has_native_streaming and hasattr(self.base_wrapper.model, 'clear_cache_encode'):
            self.base_wrapper.model.clear_cache_encode()
        else:
            self.base_wrapper.clear_cache()

    def clear_cache_decode(self):
        if self._has_native_streaming and hasattr(self.base_wrapper.model, 'clear_cache_decode'):
            self.base_wrapper.model.clear_cache_decode()
        else:
            self.base_wrapper.clear_cache()


class StreamVAEAdapter(torch.nn.Module):
    """
    Adapts factory-created VAE wrappers to StreamDiffusionV2's streaming API.

    Provides the interface expected by CausalStreamInferencePipeline:
    - .model.stream_encode() on inner model
    - .stream_decode_to_pixel() on wrapper
    - .model.first_batch flag for cache management

    If the base wrapper has native streaming methods, uses them directly for
    optimal performance. Otherwise, falls back to adapter pattern.
    """

    def __init__(self, base_vae_wrapper):
        super().__init__()
        self.base_wrapper = base_vae_wrapper
        self.model = StreamInnerModelAdapter(base_vae_wrapper)

        self._has_native_streaming = (
            hasattr(base_vae_wrapper, 'model')
            and hasattr(base_vae_wrapper.model, 'stream_decode')
        )

    def stream_decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        stream_decode_to_pixel: Wrapper-level decode method (matches vendor API).

        Uses native stream_decode if available for optimal performance, otherwise
        falls back to decode_to_pixel with caching.

        Args:
            latent: [B, T, Cz, H/8, W/8] - Compressed latent

        Returns:
            pixel: [B, T, C, H, W] - RGB frames in [-1, 1]
        """
        if self._has_native_streaming:
            zs = latent.permute(0, 2, 1, 3, 4)
            device, dtype = latent.device, latent.dtype
            scale = [
                self.base_wrapper.mean.to(device=device, dtype=dtype),
                1.0 / self.base_wrapper.std.to(device=device, dtype=dtype),
            ]
            output = self.base_wrapper.model.stream_decode(zs, scale).float().clamp_(-1, 1)
            output = output.permute(0, 2, 1, 3, 4)
            return output
        else:
            return self.base_wrapper.decode_to_pixel(latent, use_cache=True)

    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """Standard decode (for VAEInterface compatibility)"""
        return self.base_wrapper.decode_to_pixel(latent, use_cache=False)
