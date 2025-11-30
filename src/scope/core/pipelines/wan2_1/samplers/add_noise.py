"""Original add-noise sampling strategy using scheduler.add_noise()."""

import torch

from .base import Sampler


class AddNoiseSampler(Sampler):
    """
    Original sampling approach: add noise at next timestep level.

    This sampler preserves the exact original behavior of the denoising loop
    where random noise is added to the denoised prediction at the next
    timestep's noise level using scheduler.add_noise().

    This is a stateless sampler - no internal state is maintained between steps.
    """

    def reset(self) -> None:
        """Reset is a no-op for this stateless sampler."""
        pass

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
        Perform one denoising step using the original add-noise approach.

        Args:
            x: Current noisy sample [B, F, C, H, W]
            denoised: Model's denoised prediction (x0) [B, F, C, H, W]
            timestep: Current timestep tensor [B, F] (unused in this sampler)
            next_timestep: Next timestep integer value (None if final step)
            scheduler: The scheduler with add_noise method
            generator: RNG for noise generation

        Returns:
            Next noisy sample, or denoised if final step.
        """
        if next_timestep is None:
            # Final step - return the denoised prediction
            return denoised

        batch_size, num_frames = denoised.shape[:2]

        # Flatten for scheduler.add_noise which expects [B*F, C, H, W]
        flattened_pred = denoised.flatten(0, 1)

        # Generate random noise with same shape
        random_noise = torch.randn(
            flattened_pred.shape,
            device=flattened_pred.device,
            dtype=flattened_pred.dtype,
            generator=generator,
        )

        # Create timestep tensor for add_noise
        timestep_tensor = next_timestep * torch.ones(
            [batch_size * num_frames],
            device=x.device,
            dtype=torch.long,
        )

        # Add noise at next timestep level and unflatten back to [B, F, C, H, W]
        noisy = scheduler.add_noise(flattened_pred, random_noise, timestep_tensor)
        return noisy.unflatten(0, denoised.shape[:2])
