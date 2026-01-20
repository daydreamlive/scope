import torch

from scope.core.pipelines.wan2_1.vae.wan import WanVAEWrapper


# StreamDiffusionV2 does not expect the latent to be normalized, so we override the encode_to_latent method to skip that step
class StreamDiffusionV2WanVAEWrapper(WanVAEWrapper):
    def encode_to_latent(
        self, pixel: torch.Tensor, use_cache: bool = True, encoder_cache=None
    ) -> torch.Tensor:
        """Encode video pixels to latents without normalization.

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
        if use_cache:
            # Streaming encode - cache is maintained across calls
            latent = self.model.stream_encode(pixel)
        else:
            # Batch encode with one-time cache (does not affect streaming state)
            # StreamDiffusionV2 does not apply normalization, so we pass empty scale
            latent = self._encode_with_cache(
                pixel, scale=[0.0, 1.0], feat_cache=self._create_encoder_cache()
            )

        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return latent.permute(0, 2, 1, 3, 4)
