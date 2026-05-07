"""Validation that LoRA tensors must match the target model's layer shapes.

Mismatched dimensions (e.g. a Wan-14B-trained LoRA loaded against a Wan-1.3B
model) used to crash deep inside the first forward pass with an opaque tensor
error. parse_lora_weights now fails fast with a typed LoRAIncompatibleError
naming the offending layer and dimensions.
"""

import pytest
import torch

from scope.core.pipelines.wan2_1.lora.utils import (
    LoRAIncompatibleError,
    parse_lora_weights,
)


def _model_state(in_dim: int, out_dim: int) -> dict[str, torch.Tensor]:
    return {"blocks.0.self_attn.q.weight": torch.zeros(out_dim, in_dim)}


def _lora_state(in_dim: int, out_dim: int, rank: int = 8) -> dict[str, torch.Tensor]:
    return {
        "diffusion_model.blocks.0.self_attn.q.lora_A.weight": torch.zeros(rank, in_dim),
        "diffusion_model.blocks.0.self_attn.q.lora_B.weight": torch.zeros(
            out_dim, rank
        ),
    }


def test_parse_lora_weights_accepts_matching_dims():
    mapping = parse_lora_weights(_lora_state(1536, 1536), _model_state(1536, 1536))
    assert "blocks.0.self_attn.q.weight" in mapping
    assert mapping["blocks.0.self_attn.q.weight"]["rank"] == 8


def test_parse_lora_weights_rejects_in_dim_mismatch():
    # LoRA trained on Wan 14B (5120) loaded against Wan 1.3B (1536).
    with pytest.raises(LoRAIncompatibleError) as excinfo:
        parse_lora_weights(
            _lora_state(5120, 5120),
            _model_state(1536, 1536),
            lora_path="/models/lora/wan14b_style.safetensors",
        )
    msg = str(excinfo.value)
    assert "wan14b_style.safetensors" in msg
    assert "1536" in msg and "5120" in msg
    assert "blocks.0.self_attn.q.weight" in msg


def test_parse_lora_weights_rejects_out_dim_mismatch():
    # Same in_dim but different out_dim (e.g. partial dim mismatch).
    with pytest.raises(LoRAIncompatibleError):
        parse_lora_weights(_lora_state(1536, 5120), _model_state(1536, 1536))
