"""DDPM sampler for flow matching models."""

import torch

from .base import Sampler


class DDPMSampler(Sampler):
    """
    DDPM (Denoising Diffusion Probabilistic Models) sampler adapted for flow matching.

    Based on ComfyUI's DDPMSampler_step, adapted for Rectified Flow (flow matching).

    For flow matching models:
    - x_t = (1 - sigma) * x0 + sigma * noise
    - alpha = 1 - sigma (NOT alpha_cumprod = 1/(sigma^2+1) which is for Karras/EDM)
    - sigma goes from 1.0 (pure noise) to 0.0 (clean image)

    This sampler performs stochastic resampling at each step, using the denoised
    prediction as x0 and adding fresh noise scaled to the next sigma level.

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
        Perform one denoising step using DDPM for flow matching.

        For flow matching, the DDPM step uses the interpolation formula:
        - x_next = (1 - sigma_next) * denoised + sigma_next * noise_component

        Where noise_component blends the implicit noise from the current sample
        with fresh noise for stochasticity.

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

        # Get next sigma
        next_timestep_tensor = next_timestep * torch.ones(
            [batch_size * num_frames],
            device=x.device,
            dtype=torch.long,
        )
        sigma_next = self._get_sigma_for_timestep(scheduler, next_timestep_tensor)

        # Reshape for broadcasting [B, F, 1, 1, 1]
        sigma = sigma.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)
        sigma_next = sigma_next.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # DDPM posterior for flow matching:
        # We want to sample x_{t-1} given x_t and x0_pred (denoised)
        #
        # The posterior mean in flow matching can be written as an interpolation:
        # mu = (sigma_next / sigma) * x + (alpha_next - alpha * sigma_next / sigma) * denoised
        #
        # Simplifying with alpha = 1 - sigma:
        # mu = (sigma_next / sigma) * x + ((1 - sigma_next) - (1 - sigma) * sigma_next / sigma) * denoised
        # mu = (sigma_next / sigma) * x + (1 - sigma_next / sigma) * denoised
        #
        # This is just linear interpolation, same as Euler!
        sigma_ratio = sigma_next / sigma.clamp(min=1e-8)
        mu = sigma_ratio * x + (1 - sigma_ratio) * denoised

        # Add stochastic noise if not at final sigma
        if (sigma_next > 1e-8).any():
            # Sample fresh noise
            noise_fresh = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )

            # DDPM adds noise with variance related to the posterior
            # For flow matching, a simple approach is to perturb towards fresh noise
            # with magnitude proportional to the step size
            #
            # Posterior variance for flow matching:
            # var = sigma_next * (1 - sigma_next/sigma) when sigma > sigma_next
            #
            # This ensures we add more noise for larger steps
            step_ratio = (sigma - sigma_next) / sigma.clamp(min=1e-8)
            posterior_std = (sigma_next * step_ratio).clamp(min=0).sqrt()

            # Only add noise where sigma_next > 0
            noise_mask = (sigma_next > 1e-8).to(x.dtype)
            mu = mu + noise_mask * posterior_std * noise_fresh * self.s_noise

        return mu
