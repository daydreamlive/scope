"""RES Multistep sampler for flow matching models."""

import torch

from .base import Sampler


class ResMultistepSampler(Sampler):
    """
    RES Multistep sampler (Second-order Multistep Method).

    Based on ComfyUI's res_multistep / sample_res_multistep implementation.
    Reference: https://arxiv.org/pdf/2308.02157

    This is a second-order multistep method that uses the previous denoised
    prediction to achieve higher accuracy. Falls back to Euler for the first
    step when no history is available.

    Key formulas:
    - phi1(t) = (exp(t) - 1) / t
    - phi2(t) = (phi1(t) - 1) / t
    - sigma_fn(t) = exp(-t)
    - t_fn(sigma) = -log(sigma)

    For the second-order update:
    - c2 = (t_prev - t_old) / h
    - b1 = phi1(-h) - phi2(-h) / c2
    - b2 = phi2(-h) / c2
    - x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)

    This is a stateful sampler - maintains old_denoised and old_sigma_down.

    Args:
        eta: Controls ancestral sampling noise. 0.0 = deterministic. Default 0.0.
        s_noise: Noise scale multiplier. Default 1.0.
    """

    def __init__(self, eta: float = 0.0, s_noise: float = 1.0):
        self.eta = eta
        self.s_noise = s_noise
        self._old_denoised: torch.Tensor | None = None
        self._old_sigma_down: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._old_denoised = None
        self._old_sigma_down = None

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

    def _get_ancestral_step(
        self, sigma_from: torch.Tensor, sigma_to: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate sigma_down and sigma_up for ancestral sampling."""
        if self.eta == 0:
            return sigma_to, torch.zeros_like(sigma_to)

        sigma_up = torch.minimum(
            sigma_to,
            self.eta
            * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2).sqrt(),
        )
        sigma_down = (sigma_to**2 - sigma_up**2).sqrt()
        return sigma_down, sigma_up

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to t = -log(sigma)."""
        return sigma.clamp(min=1e-8).log().neg()

    def _t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t back to sigma = exp(-t)."""
        return t.neg().exp()

    def _phi1(self, t: torch.Tensor) -> torch.Tensor:
        """Compute phi1(t) = (exp(t) - 1) / t."""
        return torch.expm1(t) / t.clamp(min=1e-8)

    def _phi2(self, t: torch.Tensor) -> torch.Tensor:
        """Compute phi2(t) = (phi1(t) - 1) / t."""
        return (self._phi1(t) - 1.0) / t.clamp(min=1e-8)

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
        Perform one denoising step using RES Multistep method.

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F]
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups
            generator: RNG for noise generation (if eta > 0)

        Returns:
            Next sample after this denoising step.
        """
        if next_timestep is None:
            # Final step - return the denoised prediction
            self._old_denoised = denoised.detach().clone()
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

        # Get ancestral step sigmas
        sigma_down, sigma_up = self._get_ancestral_step(sigma, sigma_next)

        if sigma_down.abs().max() < 1e-8 or self._old_denoised is None:
            # First step or sigma_down == 0: use Euler method
            d = self._compute_derivative(x, denoised, sigma)
            dt = sigma_down - sigma
            x_next = x + d * dt
        else:
            # Second-order multistep method
            t = self._sigma_to_t(sigma)
            t_old = self._sigma_to_t(self._old_sigma_down)
            t_next = self._sigma_to_t(sigma_down)

            h = t_next - t

            # Compute c2 coefficient
            # We need t_prev which corresponds to the previous sigma before stepping
            # This is approximated using the step sizes
            c2 = (t - t_old) / h.clamp(min=1e-8)

            phi1_val = self._phi1(-h)
            phi2_val = self._phi2(-h)

            # Compute b coefficients with NaN protection
            b1 = torch.nan_to_num(phi1_val - phi2_val / c2.clamp(min=1e-8), nan=0.0)
            b2 = torch.nan_to_num(phi2_val / c2.clamp(min=1e-8), nan=0.0)

            # Second-order update
            x_next = self._t_to_sigma(h) * x + h * (
                b1 * denoised + b2 * self._old_denoised
            )

        # Add ancestral noise if eta > 0
        if self.eta > 0 and (sigma_next > 1e-8).any():
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            x_next = x_next + noise * self.s_noise * sigma_up

        # Store state for next iteration
        self._old_denoised = denoised.detach().clone()
        self._old_sigma_down = sigma_down.detach().clone()

        return x_next
