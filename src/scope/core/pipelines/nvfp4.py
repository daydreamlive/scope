# ruff: noqa: ANN001, ANN201
"""NVFP4 (E2M1) quantization utilities using comfy-kitchen.

This module provides NVFP4 quantization for transformer weights using the
comfy-kitchen library, which provides optimized CUDA kernels for Blackwell GPUs.

NVFP4 uses a two-level scaling approach:
- Per-tensor scaling: Global scale factor for the entire tensor
- Block scaling: Local scale factors for 16-element blocks

Memory Benefits:
- ~4x memory reduction for transformer weights
- Activations remain in BF16 (still the main memory bottleneck)

Performance:
- comfy-kitchen provides hardware-accelerated matmul via torch._scaled_mm
- QuantizedTensor intercepts PyTorch ops and dispatches to optimized kernels

Requirements:
- Blackwell GPU (SM >= 10.0)
- comfy-kitchen[cublas] package

References:
- comfy-kitchen: https://github.com/Comfy-Org/comfy-kitchen
- NVIDIA NVFP4: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# Minimum SM version for NVFP4 hardware acceleration
MIN_SM_VERSION = (10, 0)  # Blackwell


class WeightProxy:
    """Proxy object that provides .dtype, .device, and .shape for PEFT compatibility.

    PEFT's LoRA layers access base_layer.weight.dtype to determine input casting.
    This proxy provides that interface without requiring actual weight tensors.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device | None,
        shape: tuple[int, int],
    ):
        self.dtype = dtype
        self.device = device
        self.shape = shape

    def __repr__(self) -> str:
        return (
            f"WeightProxy(dtype={self.dtype}, device={self.device}, shape={self.shape})"
        )


# Layout name for comfy-kitchen's NVFP4 layout
NVFP4_LAYOUT = "TensorCoreNVFP4Layout"


def check_nvfp4_support() -> tuple[bool, str]:
    """Check if NVFP4 is supported on current hardware.

    Returns:
        Tuple of (is_supported, reason_if_not)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    cap = torch.cuda.get_device_capability()
    if cap < MIN_SM_VERSION:
        return (
            False,
            f"Requires SM >= {MIN_SM_VERSION[0]}.{MIN_SM_VERSION[1]} (Blackwell), "
            f"current: SM {cap[0]}.{cap[1]}",
        )

    # Check if comfy-kitchen is available
    try:
        import comfy_kitchen  # noqa: F401
    except ImportError:
        return (
            False,
            "comfy-kitchen package not installed. Install with: pip install comfy-kitchen[cublas]",
        )

    # Check if QuantizedTensor and NVFP4 layout are available
    try:
        from comfy_kitchen.tensor import (  # noqa: F401
            QuantizedTensor,
            TensorCoreNVFP4Layout,
        )
    except ImportError:
        return False, "comfy-kitchen QuantizedTensor not available"

    return True, ""


class NVFP4Linear(torch.nn.Module):
    """Linear layer with NVFP4 quantized weights using comfy-kitchen.

    This module stores weights as comfy-kitchen QuantizedTensor which
    automatically dispatches to optimized NVFP4 kernels during matmul.

    For compatibility with PEFT/LoRA, this class provides a .weight property
    that returns metadata about the quantized weights (dtype, device, shape).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight will be set via from_linear
        # We store it as a regular attribute, not a parameter
        self._quantized_weight = None

        # Store original dtype for PEFT compatibility
        # PEFT's LoRA layer accesses .weight.dtype to cast inputs
        self._weight_dtype = dtype or torch.bfloat16
        self._weight_device = device

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> WeightProxy:
        """Return a proxy object that provides .dtype and .device for PEFT compatibility.

        PEFT's LoRA layers access base_layer.weight.dtype to determine input casting.
        This proxy provides that interface without dequantizing the weights.
        """
        return WeightProxy(
            dtype=self._weight_dtype,
            device=self._weight_device,
            shape=(self.out_features, self.in_features),
        )

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> NVFP4Linear:
        """Create NVFP4Linear from a standard Linear layer."""
        from comfy_kitchen.tensor import QuantizedTensor

        nvfp4_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # Store original weight properties for PEFT compatibility
        nvfp4_linear._weight_dtype = linear.weight.dtype
        nvfp4_linear._weight_device = linear.weight.device

        # Quantize weight to NVFP4 using comfy-kitchen
        # Weight shape is (out_features, in_features)
        # from_float takes a string layout name, not the class
        weight_2d = linear.weight.data
        nvfp4_linear._quantized_weight = QuantizedTensor.from_float(
            weight_2d, NVFP4_LAYOUT
        )

        if linear.bias is not None:
            nvfp4_linear.bias = torch.nn.Parameter(linear.bias.data.clone())

        logger.debug(
            f"Quantized Linear({linear.in_features}, {linear.out_features}) to NVFP4"
        )

        return nvfp4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVFP4 quantized computation.

        comfy-kitchen's QuantizedTensor automatically dispatches to
        optimized NVFP4 kernels when both operands support it.
        """
        from comfy_kitchen.tensor import QuantizedTensor

        # Handle batched input
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        # Quantize input to NVFP4 for hardware-accelerated matmul
        # from_float takes a string layout name
        x_qt = QuantizedTensor.from_float(x, NVFP4_LAYOUT)

        # Perform quantized linear operation
        # comfy-kitchen intercepts F.linear and dispatches to scaled_mm_nvfp4
        out = torch.nn.functional.linear(x_qt, self._quantized_weight, self.bias)

        # Restore batch dimensions
        if len(orig_shape) > 2:
            out = out.view(*orig_shape[:-1], -1)

        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def quantize_model_nvfp4(
    model: torch.nn.Module,
    layer_filter: Callable[[str, torch.nn.Module], bool] | None = None,
) -> None:
    """Quantize Linear layers in a model to NVFP4 in-place.

    Args:
        model: PyTorch model to quantize
        layer_filter: Optional function (name, module) -> bool to filter layers.
                     If None, all Linear layers are quantized.
    """
    layers_to_replace = []
    skipped_lora = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Always skip LoRA adapter layers regardless of filter
            # PEFT uses lora_A and lora_B as ModuleDict containers
            # Check both original case and lowercase for safety
            if (
                "lora_A" in name
                or "lora_B" in name
                or "lora_a" in name.lower()
                or "lora_b" in name.lower()
                or ".lora_" in name.lower()
            ):
                skipped_lora.append(name)
                continue

            if layer_filter is None or layer_filter(name, module):
                layers_to_replace.append((name, module))

    if skipped_lora:
        logger.info(f"Skipped {len(skipped_lora)} LoRA adapter layers")
        logger.debug(f"Skipped LoRA layers: {skipped_lora[:5]}...")

    logger.info(f"Quantizing {len(layers_to_replace)} Linear layers to NVFP4")

    for name, module in layers_to_replace:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Replace with NVFP4Linear
        nvfp4_module = NVFP4Linear.from_linear(module)
        setattr(parent, parts[-1], nvfp4_module)

    # Free original weights
    torch.cuda.empty_cache()


def transformer_block_filter(name: str, module: torch.nn.Module) -> bool:
    """Filter function to select transformer block linear layers for quantization.

    Quantizes:
    - Attention projections (q, k, v, out)
    - MLP/FFN layers (fc1, fc2, gate, up, down)

    Excludes:
    - Embedding layers
    - Layer norms
    - Final output projections (often need full precision)
    - LoRA adapter layers (lora_A, lora_B) - these must stay as nn.Linear for PEFT

    Args:
        name: Full module name path
        module: The module instance

    Returns:
        True if the layer should be quantized
    """
    # Skip if not a Linear layer
    if not isinstance(module, torch.nn.Linear):
        return False

    name_lower = name.lower()

    # Skip LoRA adapter layers - PEFT requires these to be standard nn.Linear
    # LoRA layers are named like: base_layer.lora_A.default, base_layer.lora_B.default
    if "lora_a" in name_lower or "lora_b" in name_lower:
        return False

    # Skip embedding and output layers
    skip_patterns = [
        "embed",
        "lm_head",
        "output_proj",
        "final",
        "norm",
        "ln_",
        "layernorm",
    ]
    for pattern in skip_patterns:
        if pattern in name_lower:
            return False

    # Include attention and MLP layers
    include_patterns = [
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "qkv",
        "mlp",
        "ffn",
        "fc1",
        "fc2",
        "gate",
        "up_proj",
        "down_proj",
        "dense",
        "linear",
        "proj",
    ]

    for pattern in include_patterns:
        if pattern in name_lower:
            return True

    # Default: quantize transformer block layers
    # Check if it's inside a transformer block
    block_patterns = ["block", "layer", "transformer"]
    for pattern in block_patterns:
        if pattern in name_lower:
            return True

    return False


def wan_transformer_block_filter(name: str, module: torch.nn.Module) -> bool:
    """Filter function specifically for Wan 2.1 transformer models.

    Quantizes layers in CausalWanAttentionBlock:
    - Self-attention: q, k, v, o projections
    - Cross-attention: q, k, v, o projections (and k_img, v_img for I2V)
    - FFN: Linear layers in the feed-forward network

    Excludes:
    - Patch embedding (Conv3d, not Linear anyway)
    - Text/time embeddings
    - Head output projection
    - Normalization layers
    - LoRA adapter layers (lora_A, lora_B) - these must stay as nn.Linear for PEFT

    Args:
        name: Full module name path
        module: The module instance

    Returns:
        True if the layer should be quantized
    """
    # Skip if not a Linear layer
    if not isinstance(module, torch.nn.Linear):
        return False

    name_lower = name.lower()

    # Skip LoRA adapter layers - PEFT requires these to be standard nn.Linear
    # LoRA layers are named like: base_layer.lora_A.default, base_layer.lora_B.default
    if "lora_a" in name_lower or "lora_b" in name_lower:
        return False

    # Skip embedding layers
    skip_patterns = [
        "patch_embedding",
        "text_embedding",
        "time_embedding",
        "time_projection",
        "img_emb",  # CLIP image embedding for I2V
        "head.head",  # Final output head
        "modulation",  # Modulation parameters
    ]
    for pattern in skip_patterns:
        if pattern in name_lower:
            return False

    # Include attention layers in blocks
    if "blocks" in name_lower:
        # Self-attention projections (but not LoRA wrappers)
        if any(
            p in name_lower
            for p in ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"]
        ):
            return True
        # Cross-attention projections
        if any(
            p in name_lower
            for p in ["cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"]
        ):
            return True
        # I2V cross-attention image projections
        if any(p in name_lower for p in ["k_img", "v_img"]):
            return True
        # FFN layers
        if "ffn" in name_lower:
            return True

    return False
