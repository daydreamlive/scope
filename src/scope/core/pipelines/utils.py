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


def calculate_input_size(pipeline_file_path: str | Path) -> int:
    """Calculate input_size from model config.

    Input size is derived from the model architecture:
    input_size = num_frame_per_block * vae_temporal_downsample_factor

    This represents the number of raw video frames needed per processing chunk
    for video-to-video mode. The VAE processes frames in chunks determined by
    vae_temporal_downsample_factor (typically 4), and num_frame_per_block
    determines how many latent frames the diffusion model processes per block.

    Args:
        pipeline_file_path: Path to the pipeline's __file__ (used to locate model.yaml)

    Returns:
        Number of raw frames required per chunk for video input
    """
    model_config = load_model_config(None, pipeline_file_path)
    num_frame_per_block = getattr(model_config, "num_frame_per_block", 3)
    vae_temporal_downsample_factor = getattr(
        model_config, "vae_temporal_downsample_factor", 4
    )
    return num_frame_per_block * vae_temporal_downsample_factor
