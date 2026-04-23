"""Tests for LoRA dimension validation in parse_lora_weights.

Regression test for issue #922: a LoRA trained for Wan2.1-5B (in_features=5120)
was silently loaded into the Wan2.1-1.3B model (in_features=1536) and only
failed 156 times at inference time with an inscrutable RuntimeError.
"""

import pytest
import torch

from scope.core.pipelines.wan2_1.lora.utils import parse_lora_weights


def _make_model_state(in_features: int, out_features: int = 256) -> dict:
    """Minimal model state dict with one linear layer."""
    return {
        "blocks.0.self_attn.q.weight": torch.zeros(out_features, in_features),
    }


def _make_lora_state(rank: int, in_features: int, out_features: int = 256) -> dict:
    """Minimal PEFT-format LoRA state targeting the same layer."""
    return {
        "diffusion_model.blocks.0.self_attn.q.lora_A.weight": torch.zeros(rank, in_features),
        "diffusion_model.blocks.0.self_attn.q.lora_B.weight": torch.zeros(out_features, rank),
    }


class TestLoRADimensionValidation:
    """Verify parse_lora_weights raises a clear error on dimension mismatch."""

    def test_compatible_lora_loads_successfully(self):
        """LoRA matching the model's dimensions should parse without error."""
        model_state = _make_model_state(in_features=1536)
        lora_state = _make_lora_state(rank=32, in_features=1536)

        mapping = parse_lora_weights(lora_state, model_state)

        assert len(mapping) == 1
        key = "blocks.0.self_attn.q.weight"
        assert key in mapping
        assert mapping[key]["rank"] == 32

    def test_incompatible_lora_raises_value_error(self):
        """LoRA trained for 5B (in_features=5120) must not silently load into 1.3B (in_features=1536)."""
        model_state = _make_model_state(in_features=1536)   # 1.3B model
        lora_state = _make_lora_state(rank=32, in_features=5120)  # 5B LoRA

        with pytest.raises(ValueError, match="LoRA dimension mismatch"):
            parse_lora_weights(lora_state, model_state)

    def test_error_message_is_user_friendly(self):
        """The error message should name the layer and the dimension sizes."""
        model_state = _make_model_state(in_features=1536)
        lora_state = _make_lora_state(rank=32, in_features=5120)

        with pytest.raises(ValueError) as exc_info:
            parse_lora_weights(lora_state, model_state)

        msg = str(exc_info.value)
        assert "blocks.0.self_attn.q" in msg, "Layer name should appear in error"
        assert "5120" in msg, "LoRA in_features should appear in error"
        assert "1536" in msg, "Model in_features should appear in error"
        assert "model size" in msg.lower() or "architecture" in msg.lower(), (
            "Error should hint at model size mismatch"
        )

    def test_out_features_mismatch_also_caught(self):
        """LoRA with wrong output dimension should also be rejected."""
        model_state = _make_model_state(in_features=1536, out_features=256)
        # LoRA with matching in_features but wrong out_features
        lora_state = {
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight": torch.zeros(32, 1536),
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight": torch.zeros(512, 32),  # wrong
        }

        with pytest.raises(ValueError, match="LoRA dimension mismatch"):
            parse_lora_weights(lora_state, model_state)

    def test_compatible_5b_lora_on_5b_model(self):
        """LoRA trained for 5B on a 5B model should load fine."""
        model_state = _make_model_state(in_features=5120, out_features=5120)
        lora_state = _make_lora_state(rank=32, in_features=5120, out_features=5120)

        mapping = parse_lora_weights(lora_state, model_state)

        assert len(mapping) == 1
