import torch

from scope.core.pipelines.wan2_1.vae.wan import WanVAEWrapper


# StreamDiffusionV2 does not expect the latent to be normalized, so we override the encode_to_latent method to skip that step
class StreamDiffusionV2WanVAEWrapper(WanVAEWrapper):
    def encode_to_latent(
        self, pixel: torch.Tensor, use_cache: bool = True
    ) -> torch.Tensor:
        latent = self.model.stream_encode(pixel)
        # [batch, channels, frames, h, w] -> [batch, frames, channels, h, w]
        return latent.permute(0, 2, 1, 3, 4)
