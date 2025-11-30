"""LCM (Latent Consistency Model) sampler for flow matching models."""

import torch

from .base import Sampler


class LCMSampler(Sampler):
    """
    LCM (Latent Consistency Model) sampler.

    Based on ComfyUI's sample_lcm implementation.
    This sampler is designed for Latent Consistency Models which predict
    the final denoised output directly. The update rule is simple:
    1. Use the denoised prediction directly
    2. Add noise scaled to the next sigma level

    For flow matching, noise is added using the formula:
    x_next = (1 - sigma_next) * denoised + sigma_next * noise

    This is a stateless stochastic sampler.

    Args:
        s_noise: Noise scale multiplier. Default 1.0.
    """

    def __init__(self, s_noise: float = 1.0):
        self.s_noise = s_noise

    def reset(self) -> None:
        """Reset is a no-op for this stateless sampler."""
        pass

    def _get_sigma_for_timestep(
        self,
        scheduler,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Get sigma value for a given timestep from the scheduler."""
        scheduler.sigmas = scheduler.sigmas.to(timestep.device)
        scheduler.timesteps = scheduler.timesteps.to(timestep.device)

        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)

        timestep_id = torch.argmin(
            (scheduler.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        return scheduler.sigmas[timestep_id]

    def step(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: int | None,
        scheduler,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Perform one denoising step using LCM.

        Algorithm:
        1. Take the denoised prediction as the current clean estimate
        2. If not final step, add noise scaled to next sigma level

        For flow matching (x_t = (1-sigma)*x0 + sigma*noise):
        x_next = (1 - sigma_next) * denoised + sigma_next * noise

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F] (unused)
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups
            generator: RNG for noise generation

        Returns:
            Next sample after this denoising step.
        """
        # Start with the denoised prediction
        x_next = denoised

        if next_timestep is None:
            # Final step - just return denoised
            return x_next

        batch_size, num_frames = x.shape[:2]

        # Get next sigma
        next_timestep_tensor = next_timestep * torch.ones(
            [batch_size * num_frames],
            device=x.device,
            dtype=torch.long,
        )
        sigma_next = self._get_sigma_for_timestep(scheduler, next_timestep_tensor)

        # Reshape for broadcasting [B, F, 1, 1, 1]
        sigma_next = sigma_next.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # Add noise if sigma_next > 0
        if (sigma_next > 1e-8).any():
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )

            # Flow matching noise scaling: x_t = (1-sigma)*x0 + sigma*noise
            # x_next = (1 - sigma_next) * denoised + sigma_next * noise
            alpha_next = 1 - sigma_next
            x_next = alpha_next * denoised + sigma_next * noise * self.s_noise

        return x_next
