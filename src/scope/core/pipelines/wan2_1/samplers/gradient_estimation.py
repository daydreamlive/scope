"""Gradient estimation sampling strategy for improved convergence."""

import torch

from .base import Sampler


class GradientEstimationSampler(Sampler):
    """
    Gradient estimation sampler for improved convergence with CFG distillation.

    Based on the technique from: https://openreview.net/pdf?id=o2ND9v0CeK

    This sampler uses the CFG_PP-style formulation which directly reconstructs
    the flow matching interpolation formula rather than using Euler steps.
    This approach is more appropriate for flow matching models.

    For flow matching where x_t = (1 - sigma) * x0 + sigma * noise:
    - velocity v = (x - x0_pred) / sigma = noise - x0 (approximation)
    - x_next = x0_pred + v * sigma_next reconstructs the interpolation

    This is a stateful sampler - it maintains the previous velocity between steps.

    Args:
        gamma: Gradient estimation strength. Default 2.0.
               Higher values = stronger momentum effect.
    """

    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
        self._old_velocity: torch.Tensor | None = None
        self._step_index: int = 0

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._old_velocity = None
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

    def _compute_velocity(
        self,
        x: torch.Tensor,
        denoised: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity v = (x - x0_pred) / sigma.

        In flow matching, this approximates the velocity field v = noise - x0.
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

        Uses CFG_PP-style formulation for flow matching:
        1. Compute velocity v = (x - denoised) / sigma
        2. Reconstruct at next sigma: x_next = denoised + v * sigma_next
        3. If not first step, apply gradient estimation correction

        This directly reconstructs the flow matching interpolation formula:
        x_next = (1 - sigma_next) * x0 + sigma_next * noise

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

        # Compute velocity v = (x - denoised) / sigma
        # This is the estimated flow velocity field
        velocity = self._compute_velocity(x, denoised, sigma)

        # CFG_PP-style step: directly reconstruct at next sigma level
        # x_next = x0_pred + velocity * sigma_next
        # This equals: (1 - sigma_next) * x0 + sigma_next * noise (flow matching formula)
        x_next = denoised + velocity * sigma_next

        # Gradient estimation correction (skip on first step)
        if self._step_index >= 1 and self._old_velocity is not None:
            # Compute dt for the correction term
            dt = sigma_next - sigma
            velocity_correction = (self.gamma - 1) * (velocity - self._old_velocity)
            x_next = x_next + velocity_correction * dt

        # Store velocity for next step
        self._old_velocity = velocity.detach().clone()
        self._step_index += 1

        return x_next
