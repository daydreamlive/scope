"""Euler Ancestral RF sampler for flow matching models."""

import torch

from .base import Sampler


class EulerAncestralRFSampler(Sampler):
    """
    Euler Ancestral sampler adapted for Rectified Flow (flow matching).

    Based on ComfyUI's sample_euler_ancestral_RF implementation.
    This sampler performs an interpolation step towards denoised, then
    optionally adds stochastic noise scaled appropriately for flow matching.

    The key formula for flow matching:
    - x_t = (1 - sigma) * x0 + sigma * noise
    - Interpolation: x_next = ratio * x + (1 - ratio) * denoised
    - Noise addition with proper scaling for the (1-sigma) term

    Args:
        eta: Controls stochasticity. 0.0 = deterministic, 1.0 = full stochastic.
             Default 1.0 matches the original AddNoiseSampler behavior.
        s_noise: Noise scale multiplier. Default 1.0.
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0):
        self.eta = eta
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
        Perform one denoising step using Euler Ancestral RF.

        Algorithm (from ComfyUI sample_euler_ancestral_RF):
        1. Compute downstep ratio based on eta
        2. Interpolate: x = ratio * x + (1 - ratio) * denoised
        3. If eta > 0, add scaled noise for stochasticity

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

        # Reshape for broadcasting
        sigma = sigma.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)
        sigma_next = sigma_next.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # Compute downstep ratio based on eta
        # downstep_ratio = 1 + (sigma_next / sigma - 1) * eta
        downstep_ratio = 1 + (sigma_next / sigma.clamp(min=1e-8) - 1) * self.eta
        sigma_down = sigma_next * downstep_ratio

        # Alpha values for flow matching: alpha = 1 - sigma
        alpha_next = 1 - sigma_next
        alpha_down = 1 - sigma_down

        # Compute renoise coefficient for flow matching
        # renoise_coeff = sqrt(sigma_next^2 - sigma_down^2 * alpha_next^2 / alpha_down^2)
        renoise_coeff = (
            (
                sigma_next**2
                - sigma_down**2 * alpha_next**2 / alpha_down.clamp(min=1e-8) ** 2
            )
            .clamp(min=0)
            .sqrt()
        )

        # Euler interpolation step
        # x = ratio * x + (1 - ratio) * denoised
        sigma_down_ratio = sigma_down / sigma.clamp(min=1e-8)
        x_next = sigma_down_ratio * x + (1 - sigma_down_ratio) * denoised

        # Add stochastic noise if eta > 0
        if self.eta > 0:
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            # Scale and add noise
            x_next = (alpha_next / alpha_down.clamp(min=1e-8)) * x_next
            x_next = x_next + noise * self.s_noise * renoise_coeff

        return x_next
