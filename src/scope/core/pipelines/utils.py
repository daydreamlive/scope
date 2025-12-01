import os
from enum import Enum
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors


class Quantization(str, Enum):
    """Quantization method enumeration."""

    FP8_E4M3FN = "fp8_e4m3fn"


def load_state_dict(weights_path: str) -> dict:
    """Load weights with automatic format detection."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    if weights_path.endswith(".safetensors"):
        # Load from safetensors and convert keys
        state_dict = load_safetensors(weights_path)

    elif weights_path.endswith(".pth") or weights_path.endswith(".pt"):
        # Load from PyTorch format (assume already in correct format)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

    else:
        raise ValueError(
            f"Unsupported file format. Expected .safetensors, .pth, or .pt, got: {weights_path}"
        )

    return state_dict


def load_model_config(config, pipeline_file_path: str | Path) -> OmegaConf:
    """
    Load model configuration from config or auto-load from model.yaml.
    Args:
        config: Configuration object that may contain a model_config attribute
        pipeline_file_path: Path to the pipeline's __file__ (used to locate model.yaml)
    Returns:
        OmegaConf: The model configuration, either from config or loaded from model.yaml
    """
    model_config = getattr(config, "model_config", None)
    if not model_config:
        model_yaml_path = Path(pipeline_file_path).parent / "model.yaml"
        model_config = OmegaConf.load(model_yaml_path)
    return model_config


def validate_resolution(
    height: int,
    width: int,
    scale_factor: int,
) -> None:
    """
    Validate that resolution dimensions are divisible by the required scale factor.

    Args:
        height: Height of the resolution
        width: Width of the resolution
        scale_factor: The factor that both dimensions must be divisible by

    Raises:
        ValueError: If height or width is not divisible by scale_factor
    """
    if height % scale_factor != 0 or width % scale_factor != 0:
        adjusted_width = (width // scale_factor) * scale_factor
        adjusted_height = (height // scale_factor) * scale_factor
        raise ValueError(
            f"Invalid resolution {width}×{height}. "
            f"Both width and height must be divisible by {scale_factor} "
            f"(VAE downsample factor 8 × patch embedding downsample factor 2 = {scale_factor}). "
            f"Please adjust to a valid resolution, e.g., {adjusted_width}×{adjusted_height}."
        )
