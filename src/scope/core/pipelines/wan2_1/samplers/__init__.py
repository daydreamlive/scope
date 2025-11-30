"""Sampler strategies for diffusion denoising loops."""

from enum import Enum

from .add_noise import AddNoiseSampler
from .base import Sampler
from .euler import EulerSampler
from .euler_ancestral_rf import EulerAncestralRFSampler
from .gradient_estimation import GradientEstimationSampler
from .ipndm import IPNDMSampler


class SamplerType(str, Enum):
    """Available sampler types."""

    ADD_NOISE = "add_noise"
    GRADIENT_ESTIMATION = "gradient_estimation"
    EULER = "euler"
    EULER_ANCESTRAL_RF = "euler_ancestral_rf"
    IPNDM = "ipndm"


def create_sampler(sampler_type: SamplerType | str, **kwargs) -> Sampler:
    """
    Factory function to create samplers.

    Args:
        sampler_type: The type of sampler to create.
        **kwargs: Sampler-specific configuration.
            - For GRADIENT_ESTIMATION: gamma (float, default 2.0)
            - For EULER: s_churn (float, default 0.0), s_noise (float, default 1.0)
            - For EULER_ANCESTRAL_RF: eta (float, default 1.0), s_noise (float, default 1.0)
            - For IPNDM: max_order (int, default 4, range 1-4)

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
    elif sampler_type == SamplerType.EULER:
        s_churn = kwargs.get("s_churn", 0.0)
        s_noise = kwargs.get("s_noise", 1.0)
        return EulerSampler(s_churn=s_churn, s_noise=s_noise)
    elif sampler_type == SamplerType.EULER_ANCESTRAL_RF:
        eta = kwargs.get("eta", 1.0)
        s_noise = kwargs.get("s_noise", 1.0)
        return EulerAncestralRFSampler(eta=eta, s_noise=s_noise)
    elif sampler_type == SamplerType.IPNDM:
        max_order = kwargs.get("max_order", 4)
        return IPNDMSampler(max_order=max_order)
    else:
        raise ValueError(
            f"create_sampler: Unknown sampler type: {sampler_type}. "
            f"Valid types: {[t.value for t in SamplerType]}"
        )


def get_available_samplers() -> list[str]:
    """Get list of available sampler type names.

    Returns:
        List of sampler type string values that can be used with create_sampler().
    """
    return [sampler_type.value for sampler_type in SamplerType]


__all__ = [
    "Sampler",
    "SamplerType",
    "AddNoiseSampler",
    "EulerSampler",
    "EulerAncestralRFSampler",
    "GradientEstimationSampler",
    "IPNDMSampler",
    "create_sampler",
    "get_available_samplers",
]
