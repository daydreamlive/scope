import json
import os
from enum import Enum
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors


class Quantization(str, Enum):
    """Quantization method enumeration."""

    FP8_E4M3FN = "fp8_e4m3fn"


def load_state_dict(weights_path: str, key_prefixes: list[str] | None = None) -> dict:
    """Load weights with automatic format detection and sharding support.

    Args:
        weights_path: Path to weights file (may not exist if sharded)
        key_prefixes: Optional list of key prefixes to filter (e.g., ["vace_blocks.", "vace_patch_embedding."])
                     If provided, only loads tensors whose keys start with these prefixes.
                     This is memory-efficient for sharded models as it avoids loading unnecessary tensors.

    Returns:
        Dictionary of loaded weights
    """
    if not os.path.exists(weights_path):
        # Check if this is a sharded model (look for index file)
        weights_dir = os.path.dirname(weights_path)
        weights_name = os.path.basename(weights_path)

        # Try to find an index file for sharded models
        if weights_name.endswith(".safetensors"):
            # Derive index filename from weights filename (e.g., diffusion_pytorch_model.safetensors -> diffusion_pytorch_model.safetensors.index.json)
            index_filename = f"{weights_name}.index.json"
            index_path = os.path.join(weights_dir, index_filename)

            if os.path.exists(index_path):
                # Load sharded safetensors model
                with open(index_path) as f:
                    index_data = json.load(f)

                weight_map = index_data.get("weight_map", {})
                if not weight_map:
                    raise ValueError(
                        f"Invalid index file format: {index_path} (no weight_map)"
                    )

                # If key_prefixes provided, filter weight_map to only needed keys
                if key_prefixes:
                    filtered_weight_map = {
                        key: shard_file
                        for key, shard_file in weight_map.items()
                        if any(key.startswith(prefix) for prefix in key_prefixes)
                    }
                    weight_map = filtered_weight_map

                # Get unique shard files needed
                shard_files = set(weight_map.values())

                # Build reverse map: shard_file -> list of keys to load from it
                shard_to_keys = {}
                for key, shard_file in weight_map.items():
                    if shard_file not in shard_to_keys:
                        shard_to_keys[shard_file] = []
                    shard_to_keys[shard_file].append(key)

                # Load only needed tensors from each shard
                state_dict = {}
                for shard_file in shard_files:
                    shard_path = os.path.join(weights_dir, shard_file)
                    if not os.path.exists(shard_path):
                        raise FileNotFoundError(f"Shard file not found: {shard_path}")

                    # Load only the keys we need from this shard
                    if key_prefixes:
                        keys_to_load = shard_to_keys.get(shard_file, [])
                        if keys_to_load:
                            # Load specific keys only (memory efficient)
                            from safetensors import safe_open

                            with safe_open(
                                shard_path, framework="pt", device="cpu"
                            ) as f:
                                for key in keys_to_load:
                                    state_dict[key] = f.get_tensor(key)
                    else:
                        # Load entire shard
                        shard_dict = load_safetensors(shard_path)
                        state_dict.update(shard_dict)

                return state_dict

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
