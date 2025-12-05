"""Wan2.1 VAE implementations.

This module provides a registry-based factory for VAE instantiation,
supporting multiple VAE types (currently WanVAEWrapper, with LightVAE planned).

Usage:
    from scope.core.pipelines.wan2_1.vae import create_vae

    # Default (WanVAEWrapper)
    vae = create_vae(model_dir="wan_models")

    # Explicit type (for UI dropdown)
    vae = create_vae(model_dir="wan_models", vae_type="wan")

    # With explicit path override
    vae = create_vae(model_dir="wan_models", vae_path="/path/to/custom_vae.pth")
"""

from dataclasses import dataclass

from .wan import WanVAEWrapper


@dataclass(frozen=True)
class VAEMetadata:
    """Metadata for a VAE type (filenames, download sources)."""

    filename: str
    download_repo: str | None = None  # None = bundled with main model repo
    download_file: str | None = None  # None = no separate download needed


# Single source of truth for VAE metadata
VAE_METADATA: dict[str, VAEMetadata] = {
    "wan": VAEMetadata(
        filename="Wan2.1_VAE.pth",
        download_repo="Wan-AI/Wan2.1-T2V-1.3B",
        download_file="Wan2.1_VAE.pth",
    ),
    "lightvae": VAEMetadata(
        filename="lightvaew2_1.pth",
        download_repo="lightx2v/Autoencoders",
        download_file="lightvaew2_1.pth",
    ),
    "tae": VAEMetadata(
        filename="taew2_1.pth",
        download_repo="lightx2v/Autoencoders",
        download_file="taew2_1.pth",
    ),
}

# Registry mapping type names to VAE classes
# UI dropdowns will use these keys
VAE_REGISTRY: dict[str, type] = {
    "wan": WanVAEWrapper,
    # "lightvae": LightVAE,  # Future: add when LightVAE is implemented
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
        vae_type: VAE type from registry. Defaults to "wan".
                  This will be selectable via UI dropdown.
        vae_path: Optional explicit path override. If provided, takes
                  precedence over model_dir/model_name path construction.

    Returns:
        Initialized VAE instance

    Raises:
        ValueError: If vae_type is not in registry
    """
    vae_type = vae_type or DEFAULT_VAE_TYPE

    vae_cls = VAE_REGISTRY.get(vae_type)
    if vae_cls is None:
        available = list(VAE_REGISTRY.keys())
        raise ValueError(
            f"create_vae: Unknown VAE type '{vae_type}'. Available types: {available}"
        )

    return vae_cls(model_dir=model_dir, model_name=model_name, vae_path=vae_path)


def list_vae_types() -> list[str]:
    """Return list of available VAE types for UI dropdowns."""
    return list(VAE_REGISTRY.keys())


__all__ = [
    "WanVAEWrapper",
    "VAEMetadata",
    "VAE_METADATA",
    "VAE_REGISTRY",
    "DEFAULT_VAE_TYPE",
    "create_vae",
    "list_vae_types",
]
