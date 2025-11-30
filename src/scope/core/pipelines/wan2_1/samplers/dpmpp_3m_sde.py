"""DPM-Solver++(3M) SDE sampler for flow matching models."""

import torch

from .base import Sampler


class DPMPP3MSDESampler(Sampler):
    """
    DPM-Solver++(3M) SDE (Stochastic Differential Equation) sampler.

    Based on ComfyUI's sample_dpmpp_3m_sde implementation, adapted for flow matching.
    This is a third-order variant of DPM-Solver++ SDE that uses two previous
    denoised predictions for higher-order accuracy.

    Key formulas:
    - lambda_fn(sigma) = log((1-sigma)/sigma) for flow matching
    - h = lambda_t - lambda_s
    - h_eta = h * (eta + 1)

    First-order base:
    - x = (sigma_next/sigma) * exp(-h*eta) * x + alpha_t * (-h_eta).expm1().neg() * denoised

    Third-order correction (when 2 history points available):
    - r0 = h_1 / h, r1 = h_2 / h
    - d1_0 = (denoised - denoised_1) / r0
    - d1_1 = (denoised_1 - denoised_2) / r1
    - d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
    - d2 = (d1_0 - d1_1) / (r0 + r1)
    - phi_2 = h_eta.neg().expm1() / h_eta + 1
    - phi_3 = phi_2 / h_eta - 0.5
    - x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2

    This is a stateful stochastic sampler - keeps denoised_1, denoised_2, h_1, h_2.

    Args:
        eta: Controls stochasticity. 0.0 = deterministic, 1.0 = full SDE. Default 1.0.
        s_noise: Noise scale multiplier. Default 1.0.
    """

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0):
        self.eta = eta
        self.s_noise = s_noise
        self._denoised_1: torch.Tensor | None = None
        self._denoised_2: torch.Tensor | None = None
        self._h_1: torch.Tensor | None = None
        self._h_2: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._denoised_1 = None
        self._denoised_2 = None
        self._h_1 = None
        self._h_2 = None

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

    def _sigma_to_lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to half-logSNR lambda.

        For flow matching (CONST): lambda = log((1-sigma)/sigma) = -logit(sigma)
        """
        sigma_clamped = sigma.clamp(min=1e-8, max=1 - 1e-8)
        return ((1 - sigma_clamped) / sigma_clamped).log()

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
        Perform one denoising step using DPM-Solver++(3M) SDE.

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
            # Final step - update history and return denoised
            self._denoised_2 = self._denoised_1
            self._denoised_1 = denoised.detach().clone()
            self._h_2 = self._h_1
            self._h_1 = None
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

        # Convert to lambda (half-logSNR)
        lambda_s = self._sigma_to_lambda(sigma)
        lambda_t = self._sigma_to_lambda(sigma_next)
        h = lambda_t - lambda_s
        h_eta = h * (self.eta + 1)

        # Compute alpha_t for flow matching: alpha = 1 - sigma
        alpha_t = 1 - sigma_next

        # First-order base term
        x_next = (
            sigma_next / sigma.clamp(min=1e-8) * (-h * self.eta).exp() * x
            + alpha_t * (-h_eta).expm1().neg() * denoised
        )

        # Higher-order corrections
        if self._h_2 is not None and self._denoised_2 is not None:
            # DPM-Solver++(3M) SDE - use two history points
            r0 = self._h_1 / h.clamp(min=1e-8)
            r1 = self._h_2 / h.clamp(min=1e-8)

            d1_0 = (denoised - self._denoised_1) / r0.clamp(min=1e-8)
            d1_1 = (self._denoised_1 - self._denoised_2) / r1.clamp(min=1e-8)
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1).clamp(min=1e-8)
            d2 = (d1_0 - d1_1) / (r0 + r1).clamp(min=1e-8)

            phi_2 = h_eta.neg().expm1() / h_eta.clamp(min=1e-8) + 1
            phi_3 = phi_2 / h_eta.clamp(min=1e-8) - 0.5

            x_next = x_next + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2

        elif self._h_1 is not None and self._denoised_1 is not None:
            # DPM-Solver++(2M) SDE - use one history point
            r = self._h_1 / h.clamp(min=1e-8)
            d = (denoised - self._denoised_1) / r.clamp(min=1e-8)
            phi_2 = h_eta.neg().expm1() / h_eta.clamp(min=1e-8) + 1

            x_next = x_next + (alpha_t * phi_2) * d

        # Add SDE noise if eta > 0
        if self.eta > 0 and self.s_noise > 0 and (sigma_next > 1e-8).any():
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            # SDE noise term: sigma_next * sqrt(-expm1(-2*h*eta)) * s_noise
            noise_scale = (-2 * h * self.eta).expm1().neg().sqrt()
            x_next = x_next + noise * sigma_next * noise_scale * self.s_noise

        # Update history - shift older values
        self._denoised_2 = self._denoised_1
        self._denoised_1 = denoised.detach().clone()
        self._h_2 = self._h_1
        self._h_1 = h.detach().clone()

        return x_next
