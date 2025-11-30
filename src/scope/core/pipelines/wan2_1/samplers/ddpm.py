"""DDPM sampler for flow matching models."""

import torch

from .base import Sampler


class DDPMSampler(Sampler):
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler adapted for flow matching.

    Based on ComfyUI's sample_ddpm/DDPMSampler_step implementation.
    This sampler uses the DDPM update rule with proper alpha_cumprod scaling.

    For flow matching models, the relationship between sigma and alpha_cumprod is:
    - alpha_cumprod = 1 / (sigma^2 + 1)
    - The model predicts noise, which we convert to denoised prediction

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
        Perform one denoising step using DDPM.

        The DDPM step formula (adapted for sigma parameterization):
        - alpha_cumprod = 1 / (sigma^2 + 1)
        - alpha_cumprod_prev = 1 / (sigma_prev^2 + 1)
        - alpha = alpha_cumprod / alpha_cumprod_prev
        - noise = (x - denoised) / sigma  (the predicted noise)
        - mu = (1/sqrt(alpha)) * (x - (1-alpha) * noise / sqrt(1-alpha_cumprod))
        - If sigma_prev > 0: mu += sqrt((1-alpha)(1-alpha_cumprod_prev)/(1-alpha_cumprod)) * noise_sample

        For flow matching, we scale x appropriately.

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F]
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups
            generator: RNG for noise generation

        Returns:
            Next sample after this denoising step.
        """
        if next_timestep is None:
            # Final step - return the denoised prediction
            return denoised

        batch_size, num_frames = x.shape[:2]

        # Get current sigma
        sigma = self._get_sigma_for_timestep(scheduler, timestep.flatten(0, 1))

        # Get next sigma (sigma_prev in DDPM notation, going backward in diffusion)
        next_timestep_tensor = next_timestep * torch.ones(
            [batch_size * num_frames],
            device=x.device,
            dtype=torch.long,
        )
        sigma_prev = self._get_sigma_for_timestep(scheduler, next_timestep_tensor)

        # Reshape for broadcasting [B, F, 1, 1, 1]
        sigma = sigma.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)
        sigma_prev = sigma_prev.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # Compute alpha_cumprod values
        # For EDM/Karras parameterization: alpha_cumprod = 1 / (sigma^2 + 1)
        alpha_cumprod = 1 / (sigma**2 + 1)
        alpha_cumprod_prev = 1 / (sigma_prev**2 + 1)
        alpha = alpha_cumprod / alpha_cumprod_prev.clamp(min=1e-8)

        # Compute noise from model prediction
        # noise = (x - denoised) / sigma
        noise_pred = (x - denoised) / sigma.clamp(min=1e-8)

        # DDPM mean computation
        # mu = (1/sqrt(alpha)) * (x - (1-alpha) * noise / sqrt(1-alpha_cumprod))
        # Simplified: mu = (1/sqrt(alpha)) * (x - (1-alpha) / sqrt(1-alpha_cumprod) * noise)
        one_minus_alpha_cumprod = (1 - alpha_cumprod).clamp(min=1e-8)
        mu = (1.0 / alpha.sqrt().clamp(min=1e-8)) * (
            x - (1 - alpha) * noise_pred / one_minus_alpha_cumprod.sqrt()
        )

        # Add noise if not at final sigma
        if (sigma_prev > 1e-8).any():
            # Posterior variance: (1-alpha) * (1-alpha_cumprod_prev) / (1-alpha_cumprod)
            one_minus_alpha_cumprod_prev = (1 - alpha_cumprod_prev).clamp(min=1e-8)
            posterior_var = (
                (1 - alpha) * one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod
            )

            # Sample noise
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )

            # Add noise scaled by posterior standard deviation
            # Only add noise where sigma_prev > 0
            noise_mask = (sigma_prev > 1e-8).to(x.dtype)
            mu = mu + noise_mask * posterior_var.sqrt() * noise * self.s_noise

        return mu
