"""Sampler strategies for diffusion denoising loops."""

from enum import Enum

from .add_noise import AddNoiseSampler
from .base import Sampler
from .gradient_estimation import GradientEstimationSampler


class SamplerType(str, Enum):
    """Available sampler types."""

    ADD_NOISE = "add_noise"
    GRADIENT_ESTIMATION = "gradient_estimation"


def create_sampler(sampler_type: SamplerType | str, **kwargs) -> Sampler:
    """
    Factory function to create samplers.

    Args:
        sampler_type: The type of sampler to create.
        **kwargs: Sampler-specific configuration.
            - For GRADIENT_ESTIMATION: gamma (float, default 2.0)

    Returns:
        An instance of the requested sampler.

    Raises:
        ValueError: If sampler_type is unknown.
    """
    # Normalize string to enum
    if isinstance(sampler_type, str):
        try:
            sampler_type = SamplerType(sampler_type)
        except ValueError:
            raise ValueError(
                f"create_sampler: Unknown sampler type: {sampler_type}. "
                f"Valid types: {[t.value for t in SamplerType]}"
            ) from None

    if sampler_type == SamplerType.ADD_NOISE:
        return AddNoiseSampler()
    elif sampler_type == SamplerType.GRADIENT_ESTIMATION:
        gamma = kwargs.get("gamma", 2.0)
        return GradientEstimationSampler(gamma=gamma)
    else:
        raise ValueError(
            f"create_sampler: Unknown sampler type: {sampler_type}. "
            f"Valid types: {[t.value for t in SamplerType]}"
        )


__all__ = [
    "Sampler",
    "SamplerType",
    "AddNoiseSampler",
    "GradientEstimationSampler",
    "create_sampler",
]
