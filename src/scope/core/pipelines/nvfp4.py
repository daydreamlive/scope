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

    The weight is stored as an nn.Parameter containing a QuantizedTensor,
    which enables the __torch_dispatch__ mechanism to route F.linear calls
    to optimized NVFP4 kernels.

    For compatibility with PEFT/LoRA, this class stores the original dtype
    so LoRA adapters can properly cast inputs.
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

        # Store original dtype for PEFT compatibility and input quantization
        # PEFT's LoRA layer accesses .weight.dtype to cast inputs
        self._orig_dtype = dtype or torch.bfloat16
        self._layout_type = NVFP4_LAYOUT

        # Weight will be set via from_linear as a Parameter containing QuantizedTensor
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

        Note: This method does NOT free the original linear layer's memory.
        The caller is responsible for deleting the original module after
        this method returns.
        """
        from comfy_kitchen.tensor import QuantizedTensor

        # Capture metadata before we potentially lose access
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

        # Quantize weight to NVFP4 using comfy-kitchen
        # Weight shape is (out_features, in_features)
        # from_float takes a string layout name
        # Use .detach() to avoid keeping computation graph references
        weight_2d = linear.weight.data.detach()
        quantized_weight = QuantizedTensor.from_float(weight_2d, NVFP4_LAYOUT)

        # Store as nn.Parameter - this is critical for __torch_dispatch__ to work
        # Note: QuantizedTensor stores data internally as _qdata (uint8) which is ~4x smaller
        nvfp4_linear.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)

        # Log actual memory usage of the quantized weight
        if hasattr(quantized_weight, "_qdata"):
            qdata = quantized_weight._qdata
            logger.debug(
                f"Quantized Linear({in_features}, {out_features}): "
                f"original={weight_2d.numel() * weight_2d.element_size()} bytes, "
                f"quantized _qdata={qdata.numel() * qdata.element_size()} bytes"
            )

        if has_bias:
            # Clone bias to avoid keeping reference to original
            nvfp4_linear.bias = torch.nn.Parameter(
                linear.bias.data.detach().clone().to(dtype), requires_grad=False
            )

        return nvfp4_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVFP4 quantized computation.

        comfy-kitchen's QuantizedTensor automatically dispatches to
        optimized NVFP4 kernels when both operands support it.

        Following ComfyUI's approach:
        1. Reshape 3D input to 2D for quantization
        2. Quantize input to NVFP4
        3. Call F.linear - __torch_dispatch__ routes to scaled_mm_nvfp4
        4. Reshape output back to original dimensions
        """
        from comfy_kitchen.tensor import QuantizedTensor

        # Handle batched input - reshape 3D to 2D
        orig_shape = x.shape
        reshaped_3d = x.dim() == 3

        if reshaped_3d:
            x = x.reshape(-1, orig_shape[2])

        # Only quantize 2D tensors
        if x.dim() == 2:
            # Quantize input to NVFP4 for hardware-accelerated matmul
            x_qt = QuantizedTensor.from_float(x, self._layout_type)

            # Perform quantized linear operation
            # comfy-kitchen's __torch_dispatch__ intercepts F.linear and
            # dispatches to scaled_mm_nvfp4 when both operands are QuantizedTensor
            out = torch.nn.functional.linear(x_qt, self.weight, self.bias)
        else:
            # Fallback for non-2D tensors - dequantize weight
            weight_dq = (
                self.weight.dequantize()
                if hasattr(self.weight, "dequantize")
                else self.weight
            )
            out = torch.nn.functional.linear(x, weight_dq, self.bias)

        # Restore batch dimensions
        if reshaped_3d:
            out = out.reshape(orig_shape[0], orig_shape[1], self.weight.shape[0])

        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def quantize_model_nvfp4(
    model: torch.nn.Module,
    layer_filter: Callable[[str, torch.nn.Module], bool] | None = None,
) -> None:
    """Quantize Linear layers in a model to NVFP4 in-place.

    This function replaces nn.Linear layers with NVFP4Linear layers that use
    comfy-kitchen's QuantizedTensor for ~4x weight memory reduction and
    hardware-accelerated matmul on Blackwell GPUs.

    IMPORTANT: We only store layer NAMES (not module references) to avoid
    keeping original weights alive in memory during the replacement loop.

    Args:
        model: PyTorch model to quantize
        layer_filter: Optional function (name, module) -> bool to filter layers.
                     If None, all Linear layers are quantized.
    """
    import gc

    # CRITICAL: Only store layer names, NOT module references!
    # Storing (name, module) tuples keeps all original modules alive in memory,
    # preventing garbage collection of original weights during the loop.
    layer_names_to_replace: list[str] = []
    skipped_lora: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Always skip LoRA adapter layers regardless of filter
            # PEFT uses lora_A and lora_B as ModuleDict containers
            # The full path looks like: blocks.0.self_attn.q.lora_A.default
            name_parts = name.split(".")
            is_lora_layer = any(
                part.startswith("lora_") or part in ("lora_A", "lora_B")
                for part in name_parts
            )

            if is_lora_layer:
                skipped_lora.append(name)
                continue

            if layer_filter is None or layer_filter(name, module):
                layer_names_to_replace.append(name)

    if skipped_lora:
        logger.info(f"Skipped {len(skipped_lora)} LoRA adapter layers")
        logger.debug(f"Skipped LoRA layers: {skipped_lora[:5]}...")

    num_layers = len(layer_names_to_replace)
    logger.info(f"Quantizing {num_layers} Linear layers to NVFP4")

    # Log memory before quantization - ensure clean state first
    mem_before = 0.0
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU memory before quantization: {mem_before:.2f} GB allocated, "
            f"{mem_reserved:.2f} GB reserved"
        )

    # Process each layer by name - look up module fresh each iteration
    for i, name in enumerate(layer_names_to_replace):
        # Navigate to parent module and get the Linear layer
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Get the current module (this is the only reference we hold)
        module = getattr(parent, parts[-1])

        # Verify it's still a Linear (not already replaced)
        if not isinstance(module, torch.nn.Linear):
            logger.warning(f"Layer {name} is no longer nn.Linear, skipping")
            continue

        # Create NVFP4Linear from the original Linear
        # from_linear() creates new quantized weights, does NOT keep reference to original
        nvfp4_module = NVFP4Linear.from_linear(module)

        # Replace in the model - this removes the model's reference to original module
        setattr(parent, parts[-1], nvfp4_module)

        # Explicitly clear the original module's weight data to force CUDA memory release
        # This is necessary because 'module' still holds a reference until end of iteration
        if module.weight is not None:
            module.weight.data = torch.empty(0, device="cpu", dtype=torch.float32)
        if module.bias is not None:
            module.bias.data = torch.empty(0, device="cpu", dtype=torch.float32)

        # Delete local reference - now module can be garbage collected
        del module

        # Periodic cleanup to prevent memory fragmentation
        # More frequent cleanup (every 25 layers) for better memory behavior
        if (i + 1) % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024**3
                logger.debug(
                    f"Quantized {i + 1}/{num_layers} layers, "
                    f"current memory: {current_mem:.2f} GB"
                )

    # Final garbage collection and cache clear
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Log memory after quantization
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_reserved_after = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU memory after quantization: {mem_after:.2f} GB allocated, "
            f"{mem_reserved_after:.2f} GB reserved"
        )
        mem_saved = mem_before - mem_after
        if mem_saved > 0:
            logger.info(f"Memory saved: {mem_saved:.2f} GB")
        else:
            logger.warning(
                f"Memory INCREASED by {-mem_saved:.2f} GB - this may indicate "
                "quantized weights are larger than expected or original weights "
                "were not fully freed"
            )

        # Verify quantization by checking a sample layer
        if layer_names_to_replace:
            sample_name = layer_names_to_replace[0]
            parts = sample_name.split(".")
            sample_module = model
            for part in parts:
                sample_module = getattr(sample_module, part)

            if isinstance(sample_module, NVFP4Linear):
                weight = sample_module.weight
                logger.info(
                    f"Sample layer '{sample_name}' weight type: {type(weight).__name__}"
                )
                logger.info(
                    f"  weight.shape: {weight.shape}, weight.dtype: {weight.dtype}"
                )
                if hasattr(weight, "_qdata"):
                    qdata = weight._qdata
                    original_bytes = weight.shape[0] * weight.shape[1] * 2
                    quantized_bytes = qdata.numel() * qdata.element_size()
                    ratio = (
                        original_bytes / quantized_bytes if quantized_bytes > 0 else 0
                    )
                    logger.info(
                        f"  _qdata.shape: {qdata.shape}, _qdata.dtype: {qdata.dtype}"
                    )
                    logger.info(
                        f"  Storage: {quantized_bytes:,} bytes "
                        f"(vs {original_bytes:,} bytes for BF16, {ratio:.1f}x reduction)"
                    )
            else:
                logger.warning(
                    f"Sample layer '{sample_name}' is NOT NVFP4Linear, "
                    f"got {type(sample_module).__name__}"
                )


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
    # LoRA layers are named like: blocks.0.self_attn.q.lora_A.default
    name_parts = name.split(".")
    is_lora_layer = any(
        part.lower().startswith("lora_") or part in ("lora_A", "lora_B")
        for part in name_parts
    )
    if is_lora_layer:
        return False

    # Skip embedding, output layers, and input projection layers
    # Input projections like patchify_proj are often referenced by non-Module classes
    # and replacing them can cause stale reference issues
    skip_patterns = [
        "embed",
        "lm_head",
        "output_proj",
        "final",
        "norm",
        "ln_",
        "layernorm",
        "patchify",  # Input projection layers (patchify_proj, audio_patchify_proj)
        "caption_projection",  # Text projection layers
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
    # LoRA layers are named like: blocks.0.self_attn.q.lora_A.default
    name_parts = name.split(".")
    is_lora_layer = any(
        part.lower().startswith("lora_") or part in ("lora_A", "lora_B")
        for part in name_parts
    )
    if is_lora_layer:
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
