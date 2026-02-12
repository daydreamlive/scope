"""Quantization utilities for pipeline models.

Provides shared quantization functions used across all pipelines that support
quantization (FP8 via torchao, NVFP4 via comfy-kitchen).

NVFP4 (E2M1) provides ~4x weight memory reduction on Blackwell GPUs (SM >= 10.0)
using comfy-kitchen's QuantizedTensor and optimized CUDA kernels.

FP8 (E4M3FN) provides ~2x weight memory reduction on Ada+ GPUs (SM >= 8.9)
using torchao's dynamic activation quantization.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable

import torch

from .enums import Quantization

logger = logging.getLogger(__name__)

# ============================================================================
# NVFP4 Support
# ============================================================================

# Minimum SM version for NVFP4 hardware acceleration
MIN_NVFP4_SM_VERSION = (10, 0)  # Blackwell

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
    if cap < MIN_NVFP4_SM_VERSION:
        return (
            False,
            f"Requires SM >= {MIN_NVFP4_SM_VERSION[0]}.{MIN_NVFP4_SM_VERSION[1]} (Blackwell), "
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

    Stores weights as comfy-kitchen QuantizedTensor which automatically
    dispatches to optimized NVFP4 kernels during matmul.

    The weight is stored as an nn.Parameter containing a QuantizedTensor,
    enabling the __torch_dispatch__ mechanism to route F.linear calls
    to optimized NVFP4 kernels.
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
        self._orig_dtype = dtype or torch.bfloat16
        self._layout_type = NVFP4_LAYOUT

        self.register_parameter("weight", None)

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype or torch.bfloat16)
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> NVFP4Linear:
        """Create NVFP4Linear from a standard Linear layer.

        Note: Does NOT free the original linear layer's memory.
        The caller is responsible for cleanup after this returns.
        """
        from comfy_kitchen.tensor import QuantizedTensor

        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None
        device = linear.weight.device
        dtype = linear.weight.dtype

        nvfp4_linear = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            device=device,
            dtype=dtype,
        )

        weight_2d = linear.weight.data.detach()
        quantized_weight = QuantizedTensor.from_float(weight_2d, NVFP4_LAYOUT)
        nvfp4_linear.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)

        if has_bias:
            nvfp4_linear.bias = torch.nn.Parameter(
                linear.bias.data.detach().clone().to(dtype), requires_grad=False
            )

        return nvfp4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVFP4 quantized computation."""
        from comfy_kitchen.tensor import QuantizedTensor

        orig_shape = x.shape
        reshaped_3d = x.dim() == 3

        if reshaped_3d:
            x = x.reshape(-1, orig_shape[2])

        if x.dim() == 2:
            x_qt = QuantizedTensor.from_float(x, self._layout_type)
            out = torch.nn.functional.linear(x_qt, self.weight, self.bias)
        else:
            weight_dq = (
                self.weight.dequantize()
                if hasattr(self.weight, "dequantize")
                else self.weight
            )
            out = torch.nn.functional.linear(x, weight_dq, self.bias)

        if reshaped_3d:
            out = out.reshape(orig_shape[0], orig_shape[1], self.weight.shape[0])

        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _default_layer_filter(name: str, module: torch.nn.Module) -> bool:
    """Default filter for selecting transformer block linear layers for quantization.

    Quantizes attention projections and MLP/FFN layers.
    Excludes embedding layers, layer norms, output projections, and LoRA adapters.
    """
    if not isinstance(module, torch.nn.Linear):
        return False

    name_lower = name.lower()

    # Skip LoRA adapter layers
    name_parts = name.split(".")
    is_lora_layer = any(
        part.lower().startswith("lora_") or part in ("lora_A", "lora_B")
        for part in name_parts
    )
    if is_lora_layer:
        return False

    # Skip embedding, output, and input projection layers
    skip_patterns = [
        "embed",
        "lm_head",
        "output_proj",
        "final",
        "norm",
        "ln_",
        "layernorm",
        "patchify",
        "caption_projection",
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

    # Default: quantize layers inside transformer blocks
    block_patterns = ["block", "layer", "transformer"]
    for pattern in block_patterns:
        if pattern in name_lower:
            return True

    return False


def quantize_model_nvfp4(
    model: torch.nn.Module,
    layer_filter: Callable[[str, torch.nn.Module], bool] | None = None,
    streaming: bool = False,
    target_device: torch.device | None = None,
) -> None:
    """Quantize Linear layers in a model to NVFP4 in-place.

    Replaces nn.Linear layers with NVFP4Linear layers for ~4x weight memory
    reduction and hardware-accelerated matmul on Blackwell GPUs.

    Args:
        model: PyTorch model to quantize
        layer_filter: Optional function (name, module) -> bool to filter layers.
                     If None, uses _default_layer_filter.
        streaming: If True, use streaming mode for low-VRAM GPUs.
        target_device: Target device for quantization (only used in streaming mode).
    """
    if layer_filter is None:
        layer_filter = _default_layer_filter

    # Only store layer names to avoid keeping module references alive
    layer_names_to_replace: list[str] = []
    skipped_lora: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name_parts = name.split(".")
            is_lora_layer = any(
                part.startswith("lora_") or part in ("lora_A", "lora_B")
                for part in name_parts
            )

            if is_lora_layer:
                skipped_lora.append(name)
                continue

            if layer_filter(name, module):
                layer_names_to_replace.append(name)

    if skipped_lora:
        logger.info(f"Skipped {len(skipped_lora)} LoRA adapter layers")

    num_layers = len(layer_names_to_replace)
    logger.info(f"Quantizing {num_layers} Linear layers to NVFP4")

    if streaming and target_device is None:
        target_device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    if streaming:
        logger.info(
            f"Using streaming quantization mode (target device: {target_device})"
        )

    # Log memory before quantization
    mem_before = 0.0
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory before NVFP4 quantization: {mem_before:.2f} GB")

    for i, name in enumerate(layer_names_to_replace):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        module = getattr(parent, parts[-1])

        if not isinstance(module, torch.nn.Linear):
            continue

        if streaming:
            original_device = module.weight.device
            if original_device != target_device:
                module = module.to(target_device)
                setattr(parent, parts[-1], module)

            nvfp4_module = NVFP4Linear.from_linear(module)
            nvfp4_module = nvfp4_module.to("cpu")
            setattr(parent, parts[-1], nvfp4_module)

            if module.weight is not None:
                module.weight.data = torch.empty(0, device="cpu", dtype=torch.float32)
            if module.bias is not None:
                module.bias.data = torch.empty(0, device="cpu", dtype=torch.float32)
            del module

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            nvfp4_module = NVFP4Linear.from_linear(module)
            setattr(parent, parts[-1], nvfp4_module)

            if module.weight is not None:
                module.weight.data = torch.empty(0, device="cpu", dtype=torch.float32)
            if module.bias is not None:
                module.bias.data = torch.empty(0, device="cpu", dtype=torch.float32)
            del module

            if (i + 1) % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or (streaming and (i + 1) % 50 == 0):
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"Quantized {i + 1}/{num_layers} layers, "
                    f"GPU memory: {current_mem:.2f} GB"
                )

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_saved = mem_before - mem_after
        if mem_saved > 0:
            logger.info(f"NVFP4 quantization saved {mem_saved:.2f} GB GPU memory")


# ============================================================================
# Unified Quantization API
# ============================================================================


def apply_quantization(
    model: torch.nn.Module,
    quantization: Quantization | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Apply quantization to a model and move it to the target device.

    This is the shared entry point used by all pipelines that support quantization.
    It handles both FP8 and NVFP4 quantization methods, falling back to a simple
    device/dtype cast when quantization is None.

    Args:
        model: The model to quantize (typically the diffusion generator)
        quantization: Quantization method to apply, or None for no quantization
        device: Target device
        dtype: Target dtype (typically torch.bfloat16)

    Returns:
        The quantized model on the target device
    """
    if quantization == Quantization.FP8_E4M3FN:
        # Cast before quantization
        model = model.to(dtype=dtype)

        start = time.time()

        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            device=device,
        )

        print(f"Quantized diffusion model to fp8 in {time.time() - start:.3f}s")

    elif quantization == Quantization.NVFP4:
        supported, reason = check_nvfp4_support()
        if not supported:
            raise RuntimeError(f"NVFP4 quantization not supported: {reason}")

        # Cast to dtype first, then move to device
        model = model.to(dtype=dtype, device=device)

        start = time.time()
        quantize_model_nvfp4(model, layer_filter=_default_layer_filter)
        print(f"Quantized diffusion model to nvfp4 in {time.time() - start:.3f}s")

    else:
        model = model.to(device=device, dtype=dtype)

    return model


def apply_quantization_to_module(
    module: torch.nn.Module,
    quantization: Quantization | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    """Apply quantization to a specific module (e.g., VACE components).

    Unlike apply_quantization, this operates on sub-modules that are
    already on the correct device and doesn't return the module.

    Args:
        module: The module to quantize
        quantization: Quantization method to apply
        device: Target device
        dtype: Target dtype
    """
    if quantization is None:
        return

    if quantization == Quantization.FP8_E4M3FN:
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        quantize_(
            module,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            device=device,
        )

    elif quantization == Quantization.NVFP4:
        supported, reason = check_nvfp4_support()
        if not supported:
            logger.warning(f"NVFP4 not supported for sub-module, skipping: {reason}")
            return

        quantize_model_nvfp4(module, layer_filter=_default_layer_filter)
