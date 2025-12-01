"""DPM-Solver++(2M) SDE sampler for flow matching models."""

import torch

from .base import Sampler


class DPMPP2MSDESampler(Sampler):
    """
    DPM-Solver++(2M) SDE (Stochastic Differential Equation) sampler.

    Based on ComfyUI's sample_dpmpp_2m_sde implementation, adapted for Rectified Flow.

    For flow matching models:
    - x_t = (1 - sigma) * x0 + sigma * noise
    - alpha = 1 - sigma
    - sigma goes from 1.0 (pure noise) to 0.0 (clean image)
    - lambda (half-logSNR) = log(alpha/sigma) = log((1-sigma)/sigma)

    Key formulas:
    - h = lambda_t - lambda_s (positive, since lambda increases as sigma decreases)
    - h_eta = h * (eta + 1)
    - First-order: x = (sigma_next/sigma) * exp(-h*eta) * x + alpha_t * (-h_eta).expm1().neg() * denoised
    - Second-order correction with midpoint/heun method
    - Noise: x = x + sigma_next * sqrt(-expm1(-2*h*eta)) * s_noise * noise

    This is a stateful stochastic sampler.

    Args:
        eta: Controls stochasticity. 0.0 = deterministic DPM++(2M), 1.0 = full SDE.
             Default 1.0.
        s_noise: Noise scale multiplier. Default 1.0.
        solver_type: Either 'midpoint' or 'heun'. Default 'midpoint'.
    """

    def __init__(
        self, eta: float = 1.0, s_noise: float = 1.0, solver_type: str = "midpoint"
    ):
        if solver_type not in {"midpoint", "heun"}:
            raise ValueError(
                f"DPMPP2MSDESampler: solver_type must be 'midpoint' or 'heun', got {solver_type}"
            )
        self.eta = eta
        self.s_noise = s_noise
        self.solver_type = solver_type
        self._old_denoised: torch.Tensor | None = None
        self._h_last: torch.Tensor | None = None

    def reset(self) -> None:
        """Reset internal state for a new generation sequence."""
        self._old_denoised = None
        self._h_last = None

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
        Perform one denoising step using DPM-Solver++(2M) SDE.

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

        # Clamp sigma values to avoid numerical issues with lambda calculation
        # For flow matching: lambda = log((1-sigma)/sigma)
        # At sigma=1: lambda=-inf, at sigma=0: lambda=+inf
        # We clamp to keep lambda in a reasonable range ~[-9, 9]
        sigma_clamped = sigma.clamp(min=1e-4, max=1 - 1e-4)
        sigma_next_clamped = sigma_next.clamp(min=1e-4, max=1 - 1e-4)

        # Compute lambda (half-logSNR) for flow matching
        # lambda = log(alpha/sigma) = log((1-sigma)/sigma)
        lambda_s = ((1 - sigma_clamped) / sigma_clamped).log()
        lambda_t = ((1 - sigma_next_clamped) / sigma_next_clamped).log()

        # Step size in lambda space (positive since lambda increases as sigma decreases)
        h = lambda_t - lambda_s
        h_eta = h * (self.eta + 1)

        # Compute alpha_t from lambda_t for consistency (ComfyUI style)
        # alpha_t = sigma_next * exp(lambda_t) = sigma_next * (1-sigma_next)/sigma_next = 1-sigma_next
        # Using lambda_t.exp() ensures consistency with the lambda-based formulas
        alpha_t = sigma_next_clamped * lambda_t.exp()

        # DPM-Solver++(2M) SDE first-order term
        # x = (sigma_next/sigma) * exp(-h*eta) * x + alpha_t * (-h_eta).expm1().neg() * denoised
        exp_neg_h_eta = (-h * self.eta).exp()
        expm1_neg_h_eta = (-h_eta).expm1().neg()  # = 1 - exp(-h_eta)

        x_next = (
            sigma_next_clamped / sigma_clamped * exp_neg_h_eta * x
            + alpha_t * expm1_neg_h_eta * denoised
        )

        # Multi-step correction if we have history
        if self._old_denoised is not None and self._h_last is not None:
            r = self._h_last / h

            if self.solver_type == "heun":
                # Heun's correction
                # correction = alpha_t * (expm1(-h_eta) / (-h_eta) + 1) * (1/r) * (denoised - old_denoised)
                correction = (
                    alpha_t
                    * (expm1_neg_h_eta / h_eta.clamp(min=1e-8) + 1)
                    * (1 / r)
                    * (denoised - self._old_denoised)
                )
            else:
                # Midpoint correction
                # correction = 0.5 * alpha_t * expm1(-h_eta) * (1/r) * (denoised - old_denoised)
                correction = (
                    0.5
                    * alpha_t
                    * expm1_neg_h_eta
                    * (1 / r)
                    * (denoised - self._old_denoised)
                )

            x_next = x_next + correction

        # Add SDE noise if eta > 0
        if self.eta > 0 and self.s_noise > 0 and (sigma_next > 1e-8).any():
            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            # SDE noise term: sigma_next * sqrt(-expm1(-2*h*eta)) * s_noise
            noise_scale = (-2 * h * self.eta).expm1().neg().clamp(min=0).sqrt()
            x_next = x_next + noise * sigma_next_clamped * noise_scale * self.s_noise

        # Store state for next iteration
        self._old_denoised = denoised.detach().clone()
        self._h_last = h.detach().clone()

        return x_next
