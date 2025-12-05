"""Wan2.1 VAE implementations.

This module provides a unified VAE interface through WanVAEWrapper, which supports
both the full WanVAE and the 75% pruned LightVAE via the `use_lightvae` parameter.

Usage:
    from scope.core.pipelines.wan2_1.vae import create_vae

    # Default (full WanVAE)
    vae = create_vae(model_dir="wan_models")

    # Explicit type (for UI dropdown)
    vae = create_vae(model_dir="wan_models", vae_type="wan")

    # LightVAE (75% pruned, faster but lower quality)
    vae = create_vae(model_dir="wan_models", vae_type="lightvae")

    # With explicit path override
    vae = create_vae(model_dir="wan_models", vae_path="/path/to/custom_vae.pth")
"""

from functools import partial

from .wan import WanVAEWrapper

# Registry mapping type names to VAE factory functions
# UI dropdowns will use these keys
VAE_REGISTRY: dict[str, type] = {
    "wan": WanVAEWrapper,
    "lightvae": partial(WanVAEWrapper, use_lightvae=True),
}

DEFAULT_VAE_TYPE = "wan"


def create_vae(
    model_dir: str = "wan_models",
    model_name: str = "Wan2.1-T2V-1.3B",
    vae_type: str | None = None,
    vae_path: str | None = None,
) -> WanVAEWrapper:
    """Create VAE instance by type.

    Args:
        model_dir: Base model directory
        model_name: Model subdirectory name (e.g., "Wan2.1-T2V-1.3B")
        vae_type: VAE type ("wan" for full VAE, "lightvae" for 75% pruned).
                  Defaults to "wan". This is selectable via UI dropdown.
        vae_path: Optional explicit path override. If provided, takes
                  precedence over model_dir/model_name path construction.

    Returns:
        Initialized WanVAEWrapper instance

    Raises:
        ValueError: If vae_type is not recognized
    """
    vae_type = vae_type or DEFAULT_VAE_TYPE

    vae_factory = VAE_REGISTRY.get(vae_type)
    if vae_factory is None:
        available = list(VAE_REGISTRY.keys())
        raise ValueError(
            f"create_vae: Unknown VAE type '{vae_type}'. Available types: {available}"
        )

    return vae_factory(model_dir=model_dir, model_name=model_name, vae_path=vae_path)


def list_vae_types() -> list[str]:
    """Return list of available VAE types for UI dropdowns."""
    return list(VAE_REGISTRY.keys())


__all__ = [
    "WanVAEWrapper",
    "create_vae",
    "list_vae_types",
    "VAE_REGISTRY",
    "DEFAULT_VAE_TYPE",
]
