"""Wan2.1 VAE implementations.

This module provides a unified VAE interface through WanVAEWrapper and TAEWrapper.
WanVAEWrapper supports both the full WanVAE and the 75% pruned LightVAE via the
`use_lightvae` parameter. TAEWrapper supports both TAE and LightTAE via the
`use_lighttae` parameter.

Usage:
    from scope.core.pipelines.wan2_1.vae import create_vae

    # Default (full WanVAE)
    vae = create_vae(model_dir="wan_models")

    # Explicit type (for UI dropdown)
    vae = create_vae(model_dir="wan_models", vae_type="wan")

    # LightVAE (75% pruned, faster but lower quality)
    vae = create_vae(model_dir="wan_models", vae_type="lightvae")

    # TAE (lightweight preview encoder)
    vae = create_vae(model_dir="wan_models", vae_type="tae")

    # LightTAE (with WanVAE normalization)
    vae = create_vae(model_dir="wan_models", vae_type="lighttae")

    # With explicit path override
    vae = create_vae(model_dir="wan_models", vae_path="/path/to/custom_vae.pth")
"""

from functools import partial

from scope.core.pipelines.utils import VaeType

from .tae import TAEWrapper
from .wan import WanVAEWrapper

# Registry mapping type names to VAE factory functions
# UI dropdowns will use these keys
VAE_REGISTRY: dict[str, type] = {
    "wan": WanVAEWrapper,
    "lightvae": partial(WanVAEWrapper, use_lightvae=True),
    "tae": TAEWrapper,
    "lighttae": partial(TAEWrapper, use_lighttae=True),
}

DEFAULT_VAE_TYPE = VaeType.WAN


def create_vae(
    model_dir: str = "wan_models",
    model_name: str | None = None,
    vae_type: str | None = None,
    vae_path: str | None = None,
) -> WanVAEWrapper | TAEWrapper:
    """Create VAE instance by type.

    Args:
        model_dir: Base model directory
        model_name: Model subdirectory name. If not provided, defaults based on
                    vae_type: "Autoencoders" for tae/lighttae/lightvae,
                    "Wan2.1-T2V-1.3B" for wan.
        vae_type: VAE type ("wan" for full VAE, "lightvae" for 75% pruned,
                  "tae" for TAE, "lighttae" for LightTAE).
                  Defaults to "wan". This is selectable via UI dropdown.
        vae_path: Optional explicit path override. If provided, takes
                  precedence over model_dir/model_name path construction.

    Returns:
        Initialized WanVAEWrapper or TAEWrapper instance

    Raises:
        ValueError: If vae_type is not recognized
    """
    vae_type = vae_type or DEFAULT_VAE_TYPE

    # Determine model_name based on vae_type
    # Non-"wan" VAE types are ALWAYS from Autoencoders repo, ignore passed model_name
    if vae_type in ("tae", "lighttae", "lightvae"):
        model_name = "Autoencoders"
    elif model_name is None:
        # Default for "wan" VAE type only
        model_name = "Wan2.1-T2V-1.3B"

    vae_factory = VAE_REGISTRY.get(vae_type)
    if vae_factory is None:
        available = list(VAE_REGISTRY.keys())
        raise ValueError(
            f"create_vae: Unknown VAE type '{vae_type}'. Available types: {available}"
        )

    return vae_factory(model_dir=model_dir, model_name=model_name, vae_path=vae_path)


__all__ = [
    "WanVAEWrapper",
    "TAEWrapper",
    "create_vae",
    "VAE_REGISTRY",
    "DEFAULT_VAE_TYPE",
]
