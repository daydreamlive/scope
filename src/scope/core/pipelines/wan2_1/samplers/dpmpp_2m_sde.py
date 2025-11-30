"""DPM-Solver++(2M) SDE sampler for flow matching models."""

import torch

from .base import Sampler


class DPMPP2MSDESampler(Sampler):
    """
    DPM-Solver++(2M) SDE (Stochastic Differential Equation) sampler.

    Based on ComfyUI's sample_dpmpp_2m_sde implementation, adapted for flow matching.
    This is the stochastic variant of DPM-Solver++(2M) that adds controlled noise
    between steps for better sample diversity.

    Key formulas:
    - lambda_fn(sigma) = -log(sigma) for standard models
    - For flow matching: lambda_fn(sigma) = log((1-sigma)/sigma) = -logit(sigma)
    - h = lambda_t - lambda_s
    - h_eta = h * (eta + 1)
    - x = (sigma_next/sigma) * exp(-h*eta) * x + alpha_t * (-h_eta).expm1().neg() * denoised

    Multi-step correction (midpoint solver):
    - r = h_last / h
    - x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1/r) * (denoised - old_denoised)

    Noise addition:
    - x = x + noise * sigma_next * sqrt(-expm1(-2*h*eta)) * s_noise

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

    def _sigma_to_lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to half-logSNR lambda.

        For flow matching (CONST): lambda = log((1-sigma)/sigma) = -logit(sigma)
        For standard models: lambda = -log(sigma)

        We use the flow matching version.
        """
        # For flow matching: lambda = log((1-sigma)/sigma)
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

        # Convert to lambda (half-logSNR)
        lambda_s = self._sigma_to_lambda(sigma)
        lambda_t = self._sigma_to_lambda(sigma_next)
        h = lambda_t - lambda_s
        h_eta = h * (self.eta + 1)

        # Compute alpha_t for flow matching: alpha = 1 - sigma
        alpha_t = 1 - sigma_next

        # DPM-Solver++(2M) SDE first-order term
        x_next = (
            sigma_next / sigma.clamp(min=1e-8) * (-h * self.eta).exp() * x
            + alpha_t * (-h_eta).expm1().neg() * denoised
        )

        # Multi-step correction if we have history
        if self._old_denoised is not None and self._h_last is not None:
            r = self._h_last / h.clamp(min=1e-8)

            if self.solver_type == "heun":
                # Heun's correction
                correction = (
                    alpha_t
                    * ((-h_eta).expm1().neg() / (-h_eta).clamp(min=1e-8) + 1)
                    * (1 / r.clamp(min=1e-8))
                    * (denoised - self._old_denoised)
                )
            else:
                # Midpoint correction
                correction = (
                    0.5
                    * alpha_t
                    * (-h_eta).expm1().neg()
                    * (1 / r.clamp(min=1e-8))
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
            noise_scale = (-2 * h * self.eta).expm1().neg().sqrt()
            x_next = x_next + noise * sigma_next * noise_scale * self.s_noise

        # Store state for next iteration
        self._old_denoised = denoised.detach().clone()
        self._h_last = h.detach().clone()

        return x_next
