"""Gradient estimation sampling strategy for improved convergence."""

import torch

from .base import Sampler


class GradientEstimationSampler(Sampler):
    """
    Gradient estimation sampler for improved convergence with CFG distillation.

    Based on the technique from: https://openreview.net/pdf?id=o2ND9v0CeK

    This sampler adds a momentum-like correction term based on the difference
    between current and previous derivatives, which can improve convergence
    especially when paired with CFG distillation LoRAs.

    This is a stateful sampler - it maintains the previous derivative between steps.

    Args:
        gamma: Gradient estimation strength. Default 2.0.
               Higher values = stronger momentum effect.
    """

    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
        self._old_d: torch.Tensor | None = None
        self._step_index: int = 0

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._old_d = None
        self._step_index = 0

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

    def _to_derivative(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Convert (x, denoised, sigma) to derivative d = (x - denoised) / sigma.

        This is the Karras ODE derivative used in gradient estimation.
        """
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
        Perform one denoising step with gradient estimation correction.

        Algorithm:
        1. Compute derivative d = (x - denoised) / sigma
        2. Basic Euler step: x_next = x + d * dt
        3. If not first step, add gradient estimation: x_next += (gamma - 1) * (d - old_d) * dt

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F]
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups
            generator: Unused in this deterministic sampler

        Returns:
            Next sample after this denoising step.
        """
        if next_timestep is None:
            # Final step - return the denoised prediction
            self._step_index += 1
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

        # Reshape for broadcasting and cast to x dtype to avoid float32 upcast
        sigma = sigma.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)
        sigma_next = sigma_next.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # Compute derivative d = (x - denoised) / sigma
        d = self._to_derivative(x, denoised, sigma)

        # Compute dt (change in sigma)
        dt = sigma_next - sigma

        # Basic Euler step
        x_next = x + d * dt

        # Gradient estimation correction (skip on first step)
        if self._step_index >= 1 and self._old_d is not None:
            d_bar = (self.gamma - 1) * (d - self._old_d)
            x_next = x_next + d_bar * dt

        # Store derivative for next step
        self._old_d = d.detach().clone()
        self._step_index += 1

        return x_next
