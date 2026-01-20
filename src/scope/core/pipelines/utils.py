import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors

# Re-export enums for backwards compatibility
from .enums import Quantization as Quantization  # noqa: PLC0414
from .enums import VaeType as VaeType  # noqa: PLC0414


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
            f"Please adjust to a valid resolution, e.g., {adjusted_width}×{adjusted_height}."
        )


def parse_jsonl_prompts(file_path: str) -> list[list[str]]:
    """Parse and validate a JSONL file containing prompt sequences.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of prompt sequences (each sequence is a list of prompt strings)

    Raises:
        ValueError: If the file is invalid JSONL or doesn't follow the expected format
    """
    prompt_sequences = []
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_num}: {e}") from e

            # Validate structure
            if "prompts" not in data:
                raise ValueError(
                    f"Invalid format at line {line_num}: missing 'prompts' key"
                )

            prompts = data["prompts"]
            if not isinstance(prompts, list):
                raise ValueError(
                    f"Invalid format at line {line_num}: 'prompts' must be a list of strings"
                )

            for i, prompt in enumerate(prompts):
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"Invalid format at line {line_num}: prompt at index {i} is not a string"
                    )

            prompt_sequences.append(prompts)

    if not prompt_sequences:
        raise ValueError(f"No valid prompt sequences found in {file_path}")

    return prompt_sequences


def print_statistics(latency_measures: list[float], fps_measures: list[float]) -> None:
    """Print performance statistics."""
    print("\n=== Performance Statistics ===")
    print(
        f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
    )
    print(
        f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
    )
