"""Basic Euler sampler for deterministic sampling."""

import torch

from .base import Sampler


class EulerSampler(Sampler):
    """
    Basic Euler sampler for deterministic ODE sampling.

    Based on ComfyUI's sample_euler implementation.
    This is the simplest ODE solver: x_next = x + d * dt
    where d = (x - denoised) / sigma is the derivative.

    Optionally supports "s_churn" which adds stochastic noise before
    each step to help explore the posterior.

    This is a stateless sampler.

    Args:
        s_churn: Amount of noise to add before each step. Default 0.0 (deterministic).
                 Higher values add more stochasticity. Typical range: 0-1.
        s_noise: Noise scale multiplier. Default 1.0.
    """

    def __init__(self, s_churn: float = 0.0, s_noise: float = 1.0):
        self.s_churn = s_churn
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

    def _compute_derivative(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the ODE derivative d = (x - denoised) / sigma."""
        return (x - denoised) / sigma.clamp(min=1e-8)

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
        Perform one denoising step using basic Euler method.

        Algorithm:
        1. Optionally add noise if s_churn > 0 (increases sigma slightly)
        2. Compute derivative d = (x - denoised) / sigma
        3. Euler step: x_next = x + d * dt

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F]
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups
            generator: RNG for noise generation (used if s_churn > 0)

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

        # Apply s_churn if enabled (add noise to increase sigma)
        sigma_hat = sigma
        if self.s_churn > 0:
            # Compute gamma for noise addition
            # In ComfyUI this is clamped based on s_tmin/s_tmax, but we apply uniformly
            gamma = self.s_churn
            sigma_hat = sigma * (1 + gamma)

            # Add noise: x = x + eps * sqrt(sigma_hat^2 - sigma^2)
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            x = x + noise * self.s_noise * (sigma_hat**2 - sigma**2).clamp(min=0).sqrt()

        # Compute derivative using the (possibly increased) sigma_hat
        d = self._compute_derivative(x, denoised, sigma_hat)

        # Compute dt (using sigma_hat as the starting point)
        dt = sigma_next - sigma_hat

        # Euler step
        x_next = x + d * dt

        return x_next
