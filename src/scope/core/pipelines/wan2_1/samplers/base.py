"""Base sampler interface for diffusion sampling strategies."""

from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """
    Abstract base class for sampling strategies in diffusion models.

    Each sampler is responsible for:
    1. Computing the next sample from current state + model prediction
    2. Managing its own internal state between steps (if stateful)
    3. Resetting state for new generation sequences
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new generation sequence.

        Call this before starting a new denoising loop.
        """
        pass

    @abstractmethod
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
        Perform one denoising step.

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0)
            timestep: Current timestep tensor [B, F]
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler for sigma lookups and add_noise
            generator: Optional RNG for stochastic samplers

        Returns:
            Next sample after this denoising step. On final step, returns denoised.
        """
        pass
