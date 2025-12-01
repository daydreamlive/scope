"""Sampler strategies for diffusion denoising loops."""

from enum import Enum

from .add_noise import AddNoiseSampler
from .base import Sampler
from .ddpm import DDPMSampler
from .dpmpp_2m import DPMPP2MSampler
from .dpmpp_2m_sde import DPMPP2MSDESampler
from .dpmpp_3m_sde import DPMPP3MSDESampler
from .euler import EulerSampler
from .euler_ancestral_rf import EulerAncestralRFSampler
from .gradient_estimation import GradientEstimationSampler
from .ipndm import IPNDMSampler
from .lcm import LCMSampler


class SamplerType(str, Enum):
    """Available sampler types."""

    # Basic samplers
    ADD_NOISE = "add_noise"
    EULER = "euler"

    # Flow matching / RF samplers
    EULER_ANCESTRAL_RF = "euler_ancestral_rf"

    # Multi-step samplers
    IPNDM = "ipndm"
    DPMPP_2M = "dpmpp_2m"
    DPMPP_2M_SDE = "dpmpp_2m_sde"
    DPMPP_3M_SDE = "dpmpp_3m_sde"

    # Special samplers
    GRADIENT_ESTIMATION = "gradient_estimation"
    DDPM = "ddpm"
    LCM = "lcm"


def create_sampler(sampler_type: SamplerType | str, **kwargs) -> Sampler:
    """
    Factory function to create samplers.

    Args:
        sampler_type: The type of sampler to create.
        **kwargs: Sampler-specific configuration.

            Basic Samplers:
            - ADD_NOISE: No parameters (stateless re-noising)
            - EULER: s_churn (float, default 0.0), s_noise (float, default 1.0)

            Flow Matching / RF Samplers:
            - EULER_ANCESTRAL_RF: eta (float, default 1.0), s_noise (float, default 1.0)

            Multi-step Samplers:
            - IPNDM: max_order (int, default 4, range 1-4)
            - DPMPP_2M: No parameters (uses old_denoised history)
            - DPMPP_2M_SDE: eta (float, default 1.0), s_noise (float, default 1.0),
                           solver_type (str, 'midpoint' or 'heun', default 'midpoint')
            - DPMPP_3M_SDE: eta (float, default 1.0), s_noise (float, default 1.0)

            Special Samplers:
            - GRADIENT_ESTIMATION: gamma (float, default 2.0)
            - DDPM: s_noise (float, default 1.0)
            - LCM: s_noise (float, default 1.0)

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

    # Basic samplers
    if sampler_type == SamplerType.ADD_NOISE:
        return AddNoiseSampler()

    if sampler_type == SamplerType.EULER:
        s_churn = kwargs.get("s_churn", 0.0)
        s_noise = kwargs.get("s_noise", 1.0)
        return EulerSampler(s_churn=s_churn, s_noise=s_noise)

    # Flow matching / RF samplers
    if sampler_type == SamplerType.EULER_ANCESTRAL_RF:
        eta = kwargs.get("eta", 1.0)
        s_noise = kwargs.get("s_noise", 1.0)
        return EulerAncestralRFSampler(eta=eta, s_noise=s_noise)

    # Multi-step samplers
    if sampler_type == SamplerType.IPNDM:
        max_order = kwargs.get("max_order", 4)
        return IPNDMSampler(max_order=max_order)

    if sampler_type == SamplerType.DPMPP_2M:
        return DPMPP2MSampler()

    if sampler_type == SamplerType.DPMPP_2M_SDE:
        eta = kwargs.get("eta", 1.0)
        s_noise = kwargs.get("s_noise", 1.0)
        solver_type = kwargs.get("solver_type", "midpoint")
        return DPMPP2MSDESampler(eta=eta, s_noise=s_noise, solver_type=solver_type)

    if sampler_type == SamplerType.DPMPP_3M_SDE:
        eta = kwargs.get("eta", 1.0)
        s_noise = kwargs.get("s_noise", 1.0)
        return DPMPP3MSDESampler(eta=eta, s_noise=s_noise)

    # Special samplers
    if sampler_type == SamplerType.GRADIENT_ESTIMATION:
        gamma = kwargs.get("gamma", 2.0)
        return GradientEstimationSampler(gamma=gamma)

    if sampler_type == SamplerType.DDPM:
        s_noise = kwargs.get("s_noise", 1.0)
        return DDPMSampler(s_noise=s_noise)

    if sampler_type == SamplerType.LCM:
        s_noise = kwargs.get("s_noise", 1.0)
        return LCMSampler(s_noise=s_noise)

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
    # Base class
    "Sampler",
    # Enum and factory
    "SamplerType",
    "create_sampler",
    "get_available_samplers",
    # Basic samplers
    "AddNoiseSampler",
    "EulerSampler",
    # Flow matching / RF samplers
    "EulerAncestralRFSampler",
    # Multi-step samplers
    "IPNDMSampler",
    "DPMPP2MSampler",
    "DPMPP2MSDESampler",
    "DPMPP3MSDESampler",
    # Special samplers
    "GradientEstimationSampler",
    "DDPMSampler",
    "LCMSampler",
]
