"""Fused AdaLN Triton kernels for the LongLive pipeline.

Fuses LayerNorm (no affine) + scale/shift modulation into a single kernel,
and gate+residual into another, eliminating redundant global memory round-trips.
"""

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

import torch

if HAS_TRITON:

    @triton.jit
    def _adaln_norm_modulate_kernel(
        X_ptr,
        Scale_ptr,
        Shift_ptr,
        Out_ptr,
        C: tl.constexpr,
        frame_seqlen,
        eps,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        mod_row = row // frame_seqlen

        col_offsets = tl.arange(0, BLOCK_C)
        mask = col_offsets < C

        x_raw = tl.load(X_ptr + row * C + col_offsets, mask=mask, other=0.0)
        x = x_raw.to(tl.float32)

        mean = tl.sum(x, axis=0) / C
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / C
        inv_std = tl.rsqrt(var + eps)
        normed = x_centered * inv_std

        scale = tl.load(Scale_ptr + mod_row * C + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        shift = tl.load(Shift_ptr + mod_row * C + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        out = normed * (1.0 + scale) + shift

        tl.store(Out_ptr + row * C + col_offsets, out.to(x_raw.dtype), mask=mask)

    @triton.jit
    def _gate_residual_kernel(
        X_ptr,
        Y_ptr,
        Gate_ptr,
        Out_ptr,
        C: tl.constexpr,
        frame_seqlen,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        mod_row = row // frame_seqlen

        col_offsets = tl.arange(0, BLOCK_C)
        mask = col_offsets < C

        x_raw = tl.load(X_ptr + row * C + col_offsets, mask=mask, other=0.0)
        x = x_raw.to(tl.float32)
        y = tl.load(Y_ptr + row * C + col_offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(Gate_ptr + mod_row * C + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        out = x + y * gate

        tl.store(Out_ptr + row * C + col_offsets, out.to(x_raw.dtype), mask=mask)


def fused_adaln_norm_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    num_frames: int,
    frame_seqlen: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused LayerNorm (no affine) + AdaLN modulation.

    Args:
        x: [B, B*F*S, C] — input (will use x.view(-1, C) internally)
        scale: [B, F, 1, C] — modulation scale (1+scale applied)
        shift: [B, F, 1, C] — modulation shift
        num_frames: F
        frame_seqlen: S (tokens per frame)
        eps: LayerNorm epsilon
    Returns:
        Modulated tensor same shape as x
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused_adaln_norm_modulate")

    orig_shape = x.shape
    C = x.shape[-1]

    # Flatten to [B*F*S, C]
    x_flat = x.reshape(-1, C)
    # scale/shift from [B, F, 1, C] -> [B*F, C]
    scale_flat = scale.reshape(-1, C)
    shift_flat = shift.reshape(-1, C)

    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]
    BLOCK_C = triton.next_power_of_2(C)

    _adaln_norm_modulate_kernel[(n_rows,)](
        x_flat,
        scale_flat,
        shift_flat,
        out,
        C,
        frame_seqlen,
        eps,
        BLOCK_C,
    )

    return out.view(orig_shape)


def fused_gate_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    num_frames: int,
    frame_seqlen: int,
) -> torch.Tensor:
    """Fused gate + residual: x + y * gate with broadcasting over frame tokens.

    Args:
        x: [B, F*S, C] — residual
        y: [B, F*S, C] — attention/ffn output (already flat)
        gate: [B, F, 1, C] — gate values
        num_frames: F
        frame_seqlen: S
    Returns:
        x + y * gate, shape [B, F*S, C]
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for fused_gate_residual")

    orig_shape = x.shape
    C = x.shape[-1]

    x_flat = x.reshape(-1, C)
    y_flat = y.reshape(-1, C)
    gate_flat = gate.reshape(-1, C)

    out = torch.empty_like(x_flat)
    n_rows = x_flat.shape[0]
    BLOCK_C = triton.next_power_of_2(C)

    _gate_residual_kernel[(n_rows,)](
        x_flat,
        y_flat,
        gate_flat,
        out,
        C,
        frame_seqlen,
        BLOCK_C,
    )

    return out.view(orig_shape)
