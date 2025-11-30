"""IPNDM (Improved Pseudo Numerical Diffusion Models) sampler."""

import torch

from .base import Sampler


class IPNDMSampler(Sampler):
    """
    IPNDM sampler using Adams-Bashforth multi-step method.

    Based on ComfyUI's sample_ipndm implementation.
    This sampler uses history of previous derivatives to achieve higher-order
    accuracy without additional model evaluations per step.

    The multi-step formulas (Adams-Bashforth):
    - Order 1: x_next = x + dt * d
    - Order 2: x_next = x + dt * (3*d - d_prev) / 2
    - Order 3: x_next = x + dt * (23*d - 16*d_prev + 5*d_prev2) / 12
    - Order 4: x_next = x + dt * (55*d - 59*d_prev + 37*d_prev2 - 9*d_prev3) / 24

    This is a stateful sampler - maintains buffer of previous derivatives.

    Args:
        max_order: Maximum order of the method (1-4). Default 4.
                   Higher order = more accurate but uses more history.
    """

    def __init__(self, max_order: int = 4):
        if max_order < 1 or max_order > 4:
            raise ValueError("IPNDMSampler: max_order must be between 1 and 4")
        self.max_order = max_order
        self._buffer: list[torch.Tensor] = []
        self._step_index: int = 0

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._buffer = []
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
        Perform one denoising step using IPNDM multi-step method.

        Uses Adams-Bashforth formulas with increasing order as more
        history becomes available, up to max_order.

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

        # Reshape for broadcasting
        sigma = sigma.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)
        sigma_next = sigma_next.reshape(batch_size, num_frames, 1, 1, 1).to(x.dtype)

        # Compute current derivative
        d_cur = self._compute_derivative(x, denoised, sigma)

        # Compute dt
        dt = sigma_next - sigma

        # Determine current order (ramps up as we accumulate history)
        order = min(self.max_order, self._step_index + 1)

        # Apply Adams-Bashforth formula based on order
        if order == 1:
            # First order: simple Euler
            x_next = x + dt * d_cur
        elif order == 2:
            # Second order: uses one history point
            x_next = x + dt * (3 * d_cur - self._buffer[-1]) / 2
        elif order == 3:
            # Third order: uses two history points
            x_next = (
                x
                + dt * (23 * d_cur - 16 * self._buffer[-1] + 5 * self._buffer[-2]) / 12
            )
        else:  # order == 4
            # Fourth order: uses three history points
            x_next = (
                x
                + dt
                * (
                    55 * d_cur
                    - 59 * self._buffer[-1]
                    + 37 * self._buffer[-2]
                    - 9 * self._buffer[-3]
                )
                / 24
            )

        # Update buffer with current derivative
        if len(self._buffer) == self.max_order - 1:
            # Shift buffer and add new derivative
            for k in range(self.max_order - 2):
                self._buffer[k] = self._buffer[k + 1]
            self._buffer[-1] = d_cur.detach().clone()
        else:
            self._buffer.append(d_cur.detach().clone())

        self._step_index += 1

        return x_next
