# ruff: noqa: ANN001, ANN201, ERA001, N803, N806
"""NVFP4 (E2M1) quantization utilities for LTX2 transformer.

This module provides pure PyTorch implementation of NVFP4 quantization,
leveraging torch._scaled_mm for hardware-accelerated matmul on Blackwell GPUs.

NVFP4 uses a two-level scaling approach:
- Per-tensor scaling: Global scale factor for the entire tensor
- Block scaling: Local scale factors for 16-element blocks in a swizzled layout

References:
- NVIDIA NVFP4 documentation: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- cuBLAS block scaling layout: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
- comfy-kitchen implementation: https://github.com/Comfy-Org/comfy-kitchen
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# FP4 E2M1 format constants
F4_E2M1_MAX = 6.0
F4_E2M1_EPS = 0.5

# FP8 E4M3 format constants (used for block scales)
F8_E4M3_MAX = 448.0
F8_E4M3_EPS = 0.125

# FP32 format constants
EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = (1 << (EBITS_F32 - 1)) - 1

# Minimum SM version for NVFP4 hardware acceleration
MIN_SM_VERSION = (10, 0)  # Blackwell


# =============================================================================
# Hardware Detection
# =============================================================================


def check_nvfp4_support() -> tuple[bool, str]:
    """Check if NVFP4 is supported on current hardware.

    Returns:
        Tuple of (is_supported, reason_if_not)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"

    cap = torch.cuda.get_device_capability()
    if cap < MIN_SM_VERSION:
        return False, f"Requires SM >= {MIN_SM_VERSION[0]}.{MIN_SM_VERSION[1]} (Blackwell), current: SM {cap[0]}.{cap[1]}"

    # Check if torch._scaled_mm is available
    if not hasattr(torch, "_scaled_mm"):
        return False, "torch._scaled_mm not available (requires PyTorch 2.8+)"

    # Check if float4_e2m1fn_x2 dtype is available
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return False, "torch.float4_e2m1fn_x2 dtype not available (requires PyTorch 2.8+)"

    return True, ""


# =============================================================================
# Utility Functions
# =============================================================================


def _n_ones(n: int) -> int:
    """Return an integer with n least significant bits set to 1."""
    return (1 << n) - 1


def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _float8_round(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 E4M3 precision and back to FP32."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


# =============================================================================
# Packing/Unpacking
# =============================================================================


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into each uint8 byte.

    Input shape: (..., N) where N is even
    Output shape: (..., N//2)
    """
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0, f"Last dim must be even, got {shape[-1]}"
    uint8_data = uint8_data.contiguous().view(-1)
    packed = (uint8_data[::2] << 4) | uint8_data[1::2]
    return packed.view(*shape[:-1], shape[-1] // 2)


def unpack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into two 4-bit values each.

    Input shape: (..., N)
    Output shape: (..., N*2)
    """
    assert uint8_data.is_contiguous()
    shape = uint8_data.shape
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0x0F).to(torch.uint8)
    unpacked = torch.stack([first_elements, second_elements], dim=-1)
    return unpacked.view(*shape[:-1], shape[-1] * 2)


# =============================================================================
# cuBLAS Block Scale Layout (Swizzling)
# =============================================================================


def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """Rearrange matrix to cuBLAS tiled layout for block scales.

    cuBLAS uses a specific swizzled layout for block scaling factors.
    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        flatten: If True, return flattened tensor

    Returns:
        Rearranged tensor in cuBLAS tiled layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    if flatten:
        return rearranged.flatten()

    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Reverse the cuBLAS tiled layout back to normal (H, W) layout.

    Args:
        blocked_matrix: Swizzled tensor from cuBLAS layout
        num_rows: Desired output rows (unpadded)
        num_cols: Desired output cols (unpadded)

    Returns:
        Unswizzled tensor of shape (num_rows, num_cols)
    """
    n_row_blocks = ceil_div(num_rows, 128)
    n_col_blocks = ceil_div(num_cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)

    return unblocked[:num_rows, :num_cols]


# =============================================================================
# FP32 <-> FP4 E2M1 Conversion
# =============================================================================


def _f32_to_fp4_e2m1_unpacked(x: torch.Tensor) -> torch.Tensor:
    """Convert FP32 to FP4 E2M1 format (unpacked, one value per byte).

    Based on PyTorch AO implementation with round-to-nearest-even.

    Args:
        x: Input tensor of dtype float32

    Returns:
        Tensor of dtype uint8 with FP4 values in bits 4-7
    """
    assert x.dtype == torch.float32

    ebits, mbits = 2, 1
    exp_bias = _n_ones(ebits - 1)  # 1
    max_int = _n_ones(ebits + mbits)  # 7
    sign_mask = 1 << (ebits + mbits)  # 8

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # max_normal = 2^(3-1) * (3/2) = 6.0
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    # min_normal = 2^(1-1) = 1.0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # Save the sign
    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float32)

    # Classify values
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(~saturate_mask, x < min_normal)
    normal_mask = ~(saturate_mask | denormal_mask)

    # Denormal path
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Normal path
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    # Combine branches
    result = torch.full_like(x, max_int, dtype=torch.uint8)
    result = torch.where(denormal_mask, denormal_x, result)
    result = torch.where(normal_mask, normal_x, result)

    # Add sign back
    sign_lp = (sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)).to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    result = result | sign_lp

    return result


# =============================================================================
# NVFP4 Quantization
# =============================================================================


@dataclass
class NVFP4QuantizedWeight:
    """Container for NVFP4 quantized weight data.

    Attributes:
        qdata: Packed quantized data (uint8, two FP4 values per byte)
        per_tensor_scale: Global scale factor (float32)
        block_scales: Per-block scales in cuBLAS tiled layout (float8_e4m3fn)
        orig_shape: Original weight shape before padding
    """

    qdata: torch.Tensor
    per_tensor_scale: torch.Tensor
    block_scales: torch.Tensor
    orig_shape: tuple[int, int]


def quantize_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor | None = None,
    pad_16x: bool = True,
) -> NVFP4QuantizedWeight:
    """Quantize a 2D tensor to NVFP4 format.

    Args:
        x: Input tensor (2D, typically weight matrix)
        per_tensor_scale: Global scale factor. If None, computed automatically.
        pad_16x: If True, pad dimensions to be divisible by 16

    Returns:
        NVFP4QuantizedWeight containing quantized data and scales
    """
    if x.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {x.dim()}D")

    orig_shape = x.shape
    device = x.device

    # Compute per-tensor scale if not provided
    if per_tensor_scale is None:
        per_tensor_scale = torch.amax(x.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)
    per_tensor_scale = per_tensor_scale.to(device=device, dtype=torch.float32)

    # Handle padding
    rows, cols = x.shape
    if pad_16x:
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
    else:
        padded_rows, padded_cols = rows, cols

    block_size = 16

    # Reshape to blocks of 16 elements
    x_blocks = x.reshape(padded_rows, -1, block_size)

    # Compute per-block max absolute value
    max_abs = torch.amax(torch.abs(x_blocks), dim=-1)

    # Compute block scales
    block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
    scaled_block_scales = block_scale / per_tensor_scale
    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)

    # Total scale for quantization
    total_scale = per_tensor_scale * scaled_block_scales_fp32

    # Handle zero blocks (avoid 0/0 NaN)
    zero_scale_mask = total_scale == 0
    total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

    # Scale data
    data_scaled = x_blocks.float() / total_scale_safe.unsqueeze(-1)
    data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

    # Clamp to FP4 range
    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(padded_rows, padded_cols)

    # Convert to FP4 E2M1 and pack
    data_lp = _f32_to_fp4_e2m1_unpacked(data_scaled)
    qdata = pack_uint4(data_lp)

    # Convert block scales to cuBLAS tiled layout
    block_scales = to_blocked(scaled_block_scales_fp8.to(torch.float8_e4m3fn), flatten=False)

    return NVFP4QuantizedWeight(
        qdata=qdata,
        per_tensor_scale=per_tensor_scale,
        block_scales=block_scales,
        orig_shape=orig_shape,
    )


# =============================================================================
# NVFP4 Dequantization (for fallback/debugging)
# =============================================================================


# E2M1 lookup table for dequantization
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
).unsqueeze(1)

_E2M1_LUT_CACHE: dict[tuple, torch.Tensor] = {}


def dequantize_nvfp4(qw: NVFP4QuantizedWeight, output_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize NVFP4 weight back to floating point.

    Args:
        qw: NVFP4QuantizedWeight to dequantize
        output_dtype: Target output dtype

    Returns:
        Dequantized tensor in original shape
    """
    device = qw.qdata.device

    # Get or create LUT on correct device
    cache_key = (device, output_dtype)
    if cache_key not in _E2M1_LUT_CACHE:
        _E2M1_LUT_CACHE[cache_key] = E2M1_LUT.to(device, output_dtype)
    lut = _E2M1_LUT_CACHE[cache_key]

    # Unpack FP4 values
    lo = qw.qdata & 0x0F
    hi = qw.qdata >> 4
    unpacked = torch.stack([hi, lo], dim=-1).view(qw.qdata.shape[0], -1)

    # Lookup dequantized values
    out = torch.nn.functional.embedding(unpacked.int(), lut).squeeze(-1)

    # Get dimensions
    padded_rows, padded_cols = out.shape
    block_size = 16
    num_blocks_per_row = padded_cols // block_size

    # Reshape to blocks
    out_blocks = out.reshape(padded_rows, -1, block_size)

    # Unswizzle block scales
    block_scales_unswizzled = from_blocked(qw.block_scales, padded_rows, num_blocks_per_row)

    # Compute total scale and apply
    total_scale = qw.per_tensor_scale.to(output_dtype) * block_scales_unswizzled.to(output_dtype)
    data_dequantized = out_blocks * total_scale.unsqueeze(-1)

    # Reshape and slice to original shape
    result = data_dequantized.view(padded_rows, padded_cols).to(output_dtype)
    return result[: qw.orig_shape[0], : qw.orig_shape[1]]


# =============================================================================
# NVFP4 Matrix Multiplication
# =============================================================================


def scaled_mm_nvfp4(
    input_qw: NVFP4QuantizedWeight,
    weight_qw: NVFP4QuantizedWeight,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Matrix multiplication with NVFP4 quantized inputs.

    Computes: y = (input @ weight.T) * (scale_a * scale_b) + bias

    Uses torch._scaled_mm for hardware-accelerated computation on Blackwell GPUs.

    Args:
        input_qw: Quantized input (M, K)
        weight_qw: Quantized weight (N, K) - will be transposed
        bias: Optional bias vector (N,)
        out_dtype: Output dtype

    Returns:
        Result tensor of shape (M, N)
    """
    alpha = input_qw.per_tensor_scale * weight_qw.per_tensor_scale

    # View as float4_e2m1fn_x2 for torch._scaled_mm
    a = input_qw.qdata.view(torch.float4_e2m1fn_x2)
    b = weight_qw.qdata.view(torch.float4_e2m1fn_x2).t()

    result = torch._scaled_mm(
        a,
        b,
        input_qw.block_scales.view(-1),
        weight_qw.block_scales.view(-1),
        bias=None,  # Add bias separately for better numerical stability
        out_dtype=out_dtype,
    )

    # Apply per-tensor scale
    result = result * alpha.to(out_dtype)

    # Slice to original shape
    orig_m = input_qw.orig_shape[0]
    orig_n = weight_qw.orig_shape[0]  # weight is (out_features, in_features)
    result = result[:orig_m, :orig_n]

    if bias is not None:
        result = result + bias

    return result


# =============================================================================
# Linear Layer Replacement
# =============================================================================


class NVFP4Linear(torch.nn.Module):
    """Linear layer with NVFP4 quantized weights.

    This module stores weights in NVFP4 format and performs quantized
    matmul during forward pass. Input is quantized on-the-fly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Placeholders for quantized weight
        self.register_buffer("qdata", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("per_tensor_scale", torch.empty(0, dtype=torch.float32))
        self.register_buffer("block_scales", torch.empty(0, dtype=torch.float8_e4m3fn))
        self.register_buffer("orig_shape", torch.empty(0, dtype=torch.int64))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear) -> "NVFP4Linear":
        """Create NVFP4Linear from a standard Linear layer."""
        nvfp4_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )

        # Quantize weight
        qw = quantize_nvfp4(linear.weight.data)
        nvfp4_linear.qdata = qw.qdata
        nvfp4_linear.per_tensor_scale = qw.per_tensor_scale
        nvfp4_linear.block_scales = qw.block_scales
        nvfp4_linear.orig_shape = torch.tensor(qw.orig_shape, dtype=torch.int64)

        if linear.bias is not None:
            nvfp4_linear.bias = torch.nn.Parameter(linear.bias.data.clone())

        return nvfp4_linear

    def _get_weight_qw(self) -> NVFP4QuantizedWeight:
        """Reconstruct NVFP4QuantizedWeight from buffers."""
        return NVFP4QuantizedWeight(
            qdata=self.qdata,
            per_tensor_scale=self.per_tensor_scale,
            block_scales=self.block_scales,
            orig_shape=tuple(self.orig_shape.tolist()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with NVFP4 quantized computation.

        Input is quantized on-the-fly, then matmul is performed using
        torch._scaled_mm on Blackwell GPUs.
        """
        # Handle batched input
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        # Quantize input on-the-fly
        input_qw = quantize_nvfp4(x)

        # Get weight
        weight_qw = self._get_weight_qw()

        # Perform quantized matmul
        out = scaled_mm_nvfp4(input_qw, weight_qw, self.bias, out_dtype=x.dtype)

        # Restore batch dimensions
        if len(orig_shape) > 2:
            out = out.view(*orig_shape[:-1], -1)

        return out


# =============================================================================
# Model Quantization
# =============================================================================


def quantize_model_nvfp4(
    model: torch.nn.Module,
    layer_filter=None,
) -> torch.nn.Module:
    """Quantize all Linear layers in a model to NVFP4 format.

    Args:
        model: The model to quantize
        layer_filter: Optional filter function (name, module) -> bool
                     Returns True if layer should be quantized

    Returns:
        The quantized model (modified in-place)
    """
    supported, reason = check_nvfp4_support()
    if not supported:
        raise RuntimeError(f"NVFP4 not supported: {reason}")

    quantized_count = 0
    skipped_count = 0

    # Collect modules to replace (can't modify during iteration)
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if layer_filter is not None and not layer_filter(name, module):
                skipped_count += 1
                continue

            try:
                nvfp4_linear = NVFP4Linear.from_linear(module)
                replacements.append((name, nvfp4_linear))
                quantized_count += 1
            except Exception as e:
                logger.warning(f"Failed to quantize {name}: {e}")
                skipped_count += 1

    # Apply replacements
    for name, new_module in replacements:
        # Navigate to parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    logger.info(f"NVFP4 quantization: {quantized_count} layers quantized, {skipped_count} skipped")
    return model


# Filter for transformer blocks (matches FP8 pattern in model_configurator.py)
def transformer_block_filter(name: str, module: torch.nn.Module) -> bool:
    """Filter to only quantize transformer block layers (matching FP8 behavior)."""
    quantizable_suffixes = [
        ".to_q",
        ".to_k",
        ".to_v",
        ".to_out.0",
        ".ff.net.0.proj",
        ".ff.net.2",
    ]
    return any(name.endswith(suffix) for suffix in quantizable_suffixes)
