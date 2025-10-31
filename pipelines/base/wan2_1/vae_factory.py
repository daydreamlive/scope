"""
VAE Factory for creating VAE wrappers across all pipelines.

Provides a centralized way to instantiate VAE models with automatic checkpoint
resolution and downloading.
"""

import os
from pathlib import Path
from typing import Dict, Any

from .wrapper import WanVAEWrapper
from .lightvae_wrapper import LightVAEWrapper


VAE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "wan2.1": {
        "wrapper_class": WanVAEWrapper,
        "checkpoint": "Wan2.1_VAE.pth",
        "repo_id": "lightx2v/Autoencoders",
        "description": "Standard Wan2.1 VAE (full model)",
    },
    "lightvae2.1": {
        "wrapper_class": LightVAEWrapper,
        "checkpoint": "lightvaew2_1.pth",
        "repo_id": "lightx2v/Autoencoders",
        "description": "LightVAE 2.1 (75% pruned, lower VRAM)",
    },
}


def get_available_vaes() -> Dict[str, str]:
    """
    Get dictionary of available VAE models and their descriptions.

    Returns:
        Dict mapping VAE model names to descriptions
    """
    return {name: config["description"] for name, config in VAE_CONFIGS.items()}


def _resolve_vae_path(checkpoint_filename: str, model_dir: str = None) -> Path:
    """
    Resolve the full path to a VAE checkpoint file.

    Args:
        checkpoint_filename: Name of the checkpoint file
        model_dir: Base models directory (defaults to ~/.daydream-scope/models)

    Returns:
        Path to the checkpoint file
    """
    if model_dir is None:
        model_dir = os.path.expanduser("~/.daydream-scope/models")

    base_path = Path(model_dir) / "Wan2.1-T2V-1.3B"
    return base_path / checkpoint_filename


def _download_vae_if_missing(vae_path: Path, repo_id: str, checkpoint_filename: str) -> None:
    """
    Download VAE checkpoint if it doesn't exist locally.

    Args:
        vae_path: Full path where checkpoint should exist
        repo_id: HuggingFace repo ID to download from
        checkpoint_filename: Name of the checkpoint file in the repo
    """
    if not vae_path.exists():
        try:
            from download_models import download_hf_single_file

            dst_dir = vae_path.parent
            print(f"create_vae_wrapper: downloading '{checkpoint_filename}' from {repo_id} to '{dst_dir}'")
            download_hf_single_file(repo_id=repo_id, filename=checkpoint_filename, local_dir=dst_dir)
        except ImportError:
            raise RuntimeError(
                f"create_vae_wrapper: VAE checkpoint not found at {vae_path} and download_models module not available"
            )


def create_vae_wrapper(vae_model: str = "wan2.1", model_dir: str = None):
    """
    Factory function to create a VAE wrapper based on model name.

    This function:
    1. Validates the VAE model name
    2. Resolves the checkpoint path
    3. Auto-downloads the checkpoint if missing
    4. Instantiates and returns the appropriate wrapper

    Args:
        vae_model: VAE model identifier (e.g., "wan2.1", "lightvae2.1")
        model_dir: Base directory for models (defaults to ~/.daydream-scope/models)

    Returns:
        VAE wrapper instance (WanVAEWrapper or LightVAEWrapper)

    Raises:
        ValueError: If vae_model is not recognized
        FileNotFoundError: If checkpoint cannot be found or downloaded

    Examples:
        >>> vae = create_vae_wrapper("lightvae2.1")
        >>> vae = create_vae_wrapper("wan2.1", model_dir="/path/to/models")
    """
    if vae_model not in VAE_CONFIGS:
        available = ", ".join(f"'{name}'" for name in VAE_CONFIGS.keys())
        raise ValueError(
            f"create_vae_wrapper: Unknown VAE model '{vae_model}'. "
            f"Available models: {available}"
        )

    config = VAE_CONFIGS[vae_model]
    wrapper_class = config["wrapper_class"]
    checkpoint_filename = config["checkpoint"]
    repo_id = config["repo_id"]

    vae_path = _resolve_vae_path(checkpoint_filename, model_dir)

    _download_vae_if_missing(vae_path, repo_id, checkpoint_filename)

    if not vae_path.exists():
        raise FileNotFoundError(
            f"create_vae_wrapper: VAE checkpoint not found at {vae_path} after download attempt"
        )

    print(f"create_vae_wrapper: Using {vae_model} VAE from {vae_path}")

    if wrapper_class == LightVAEWrapper:
        return LightVAEWrapper(vae_path=str(vae_path))
    elif wrapper_class == WanVAEWrapper:
        return WanVAEWrapper(model_dir=model_dir, vae_path=str(vae_path))
    else:
        raise RuntimeError(f"create_vae_wrapper: Unknown wrapper class {wrapper_class}")
