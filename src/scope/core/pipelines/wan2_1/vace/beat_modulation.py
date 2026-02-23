"""Beat-reactive modulation for VACE conditioning frames.

Applied inside VaceEncodingBlock._encode_with_conditioning() right before
vace_encode_frames(), so the latency between modulation and display is just
the VACE forward pass + denoise + decode — consistent and predictable.

Tensor format at injection point:
    input_frames_list: List[Tensor[C=3, T=12, H, W]]
    dtype: bfloat16 (VAE dtype), range [-1, 1]

All curve/effect/phase logic is self-contained here to keep
vace_encoding.py clean (single function call insertion).
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Beat curve functions  (phase 0 = on-beat, 1 = just before next beat)
# ---------------------------------------------------------------------------

_EMA_ALPHA = 0.3  # chunk-duration EMA smoothing factor


def _pulse(phase: float, decay: float = 5.0) -> float:
    return math.exp(-decay * phase)


def _sine(phase: float) -> float:
    return (math.cos(2.0 * math.pi * phase) + 1.0) / 2.0


def _square(phase: float, duty: float = 0.5) -> float:
    return 1.0 if phase < duty else 0.0


def _sawtooth(phase: float) -> float:
    return 1.0 - phase


def _triangle(phase: float) -> float:
    return 1.0 - 2.0 * abs(phase - 0.5)


_CURVES = {
    "pulse": _pulse,
    "sine": _sine,
    "square": _square,
    "sawtooth": _sawtooth,
    "triangle": _triangle,
}


def _get_curve_value(curve_name: str, phase: float) -> float:
    fn = _CURVES.get(curve_name, _sine)
    return fn(phase)


# ---------------------------------------------------------------------------
# Effect functions — adapted for [C, T, H, W] tensors in [-1, 1]
# ---------------------------------------------------------------------------


def _apply_intensity(
    frames: torch.Tensor, amount: float, beat_vals: torch.Tensor
) -> torch.Tensor:
    """Fade conditioning toward zero on off-beat.

    Lerps toward zero rather than just scaling brightness — this disrupts
    the conditioning signal structurally, which the diffusion model can't
    ignore the way it can ignore simple brightness shifts.
    """
    # beat_vals shape: [T, 1, 1] → broadcasts over C, H, W
    # On-beat (beat_val=1): mix=0 → full conditioning
    # Off-beat (beat_val=0): mix=amount → faded toward zero
    mix = amount * (1.0 - beat_vals)
    return frames * (1.0 - mix)


def _apply_invert(
    frames: torch.Tensor, amount: float, beat_vals: torch.Tensor
) -> torch.Tensor:
    """Invert conditioning on-beat.  In [-1, 1] range, inversion is negation."""
    mix = amount * beat_vals
    return torch.lerp(frames, -frames, mix)


def _apply_contrast(
    frames: torch.Tensor, amount: float, beat_vals: torch.Tensor
) -> torch.Tensor:
    """Boost contrast on-beat.  [-1, 1] is already zero-centered.

    Uses a higher multiplier (5x) so the effect is visible through the
    VACE encode → denoise → decode chain. Conditioning contrast needs to
    be extreme to produce noticeable output changes.
    """
    gain = 1.0 + 5.0 * amount * beat_vals
    return (frames * gain).clamp(-1, 1)


def _apply_blur(
    frames: torch.Tensor, amount: float, beat_vals_1d: torch.Tensor
) -> torch.Tensor:
    """Blur off-beat, sharp on-beat via downscale/upscale.

    *frames* is [C, T, H, W].  We process each frame independently.
    *beat_vals_1d* is a [T] tensor (scalar per frame).
    """
    C, T, H, W = frames.shape
    result = frames.clone()

    for i in range(T):
        bv = beat_vals_1d[i].item()
        blur_strength = amount * (1.0 - bv)
        if blur_strength < 0.01:
            continue

        # More aggressive downscale — up to 10% of original size at full amount
        scale = max(1.0 - blur_strength * 0.9, 0.1)
        small_h = max(int(H * scale), 1)
        small_w = max(int(W * scale), 1)

        f = frames[:, i, :, :].unsqueeze(0)  # [1, C, H, W]
        f = F.interpolate(f, size=(small_h, small_w), mode="bilinear", align_corners=False)
        f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        result[:, i, :, :] = f.squeeze(0)

    return result


# ---------------------------------------------------------------------------
# Phase computation
# ---------------------------------------------------------------------------


def _compute_chunk_phases(
    now: float,
    num_frames: int,
    bpm: float,
    phase_offset: float,
    timing_mode: str,
    target_fps: float,
    state: dict[str, Any],
) -> list[float]:
    """Return a beat phase [0, 1) for each frame in the chunk.

    **clock** (default): Phase derived from wall time at chunk boundary.
    Within-chunk spread uses EMA of recent chunk durations.

    **counter**: Deterministic — phase is a function of frame count and
    target_fps.  Perceived BPM = actual_output_fps * set_bpm / target_fps.

    *state* is a mutable dict that persists across chunks on the block
    instance.  Keys: ``last_time``, ``chunk_dt_ema``, ``frame_counter``.
    """
    beat_period = 60.0 / max(bpm, 1.0)

    if timing_mode == "counter":
        frame_counter = state.get("frame_counter", 0)
        phases: list[float] = []
        for i in range(num_frames):
            idx = frame_counter + i
            raw = (idx * bpm) / (60.0 * max(target_fps, 1.0))
            phases.append((raw + phase_offset) % 1.0)
        state["last_time"] = now
        state["frame_counter"] = frame_counter + num_frames
        return phases

    # --- Clock mode (default) -----------------------------------------------

    start_phase = (now / beat_period) % 1.0

    last_time = state.get("last_time")
    chunk_dt_ema = state.get("chunk_dt_ema")

    if last_time is not None:
        raw_dt = max(min(now - last_time, 3.0), 0.01)
        if chunk_dt_ema is None:
            chunk_dt_ema = raw_dt
        else:
            chunk_dt_ema += _EMA_ALPHA * (raw_dt - chunk_dt_ema)
    else:
        chunk_dt_ema = num_frames / max(target_fps, 1.0)

    chunk_phase_span = chunk_dt_ema / beat_period

    phases = []
    for i in range(num_frames):
        frac = i / max(num_frames - 1, 1)
        p = start_phase + frac * chunk_phase_span
        phases.append((p + phase_offset) % 1.0)

    state["last_time"] = now
    state["chunk_dt_ema"] = chunk_dt_ema
    state["frame_counter"] = state.get("frame_counter", 0) + num_frames
    return phases


# ---------------------------------------------------------------------------
# Main entry point — called from VaceEncodingBlock
# ---------------------------------------------------------------------------


def modulate_conditioning_frames(
    input_frames_list: list[torch.Tensor],
    block_state: Any,
    beat_state: dict[str, Any],
) -> list[torch.Tensor]:
    """Apply beat-reactive modulation to conditioning frames.

    Parameters
    ----------
    input_frames_list : list[Tensor[C, T, H, W]]
        One tensor per batch item, dtype bfloat16, range [-1, 1].
    block_state : PipelineState
        Provides all beat params via attribute access.
    beat_state : dict
        Mutable dict persisted on the block instance across chunks.
        Stores timing state (last_time, chunk_dt_ema, frame_counter).

    Returns
    -------
    list[Tensor[C, T, H, W]]
        Modulated frames (same dtype/device as input).
    """
    now = time.time()

    # Read params from block_state (set via kwargs → PipelineState)
    bpm = getattr(block_state, "bpm", 120.0)
    phase_offset = getattr(block_state, "beat_phase_offset", 0.0)
    curve_name = getattr(block_state, "beat_curve", "pulse")
    timing_mode = getattr(block_state, "timing_mode", "clock")
    target_fps = getattr(block_state, "target_fps", 15.0)
    reset_phase = getattr(block_state, "reset_phase", False)

    intensity_on = getattr(block_state, "intensity_enabled", True)
    intensity_amt = getattr(block_state, "intensity_amount", 0.8)
    blur_on = getattr(block_state, "blur_enabled", False)
    blur_amt = getattr(block_state, "blur_amount", 0.7)
    invert_on = getattr(block_state, "invert_enabled", False)
    invert_amt = getattr(block_state, "invert_amount", 0.5)
    contrast_on = getattr(block_state, "contrast_enabled", False)
    contrast_amt = getattr(block_state, "contrast_amount", 0.7)

    # Phase reset
    if reset_phase:
        beat_state.clear()

    # Check if any effect is actually enabled
    any_effect = (
        (intensity_on and intensity_amt > 0)
        or (blur_on and blur_amt > 0)
        or (invert_on and invert_amt > 0)
        or (contrast_on and contrast_amt > 0)
    )
    if not any_effect:
        return input_frames_list

    # Get frame count from first batch item
    num_frames = input_frames_list[0].shape[1]  # [C, T, H, W]

    # Compute per-frame beat phases
    phases = _compute_chunk_phases(
        now, num_frames, bpm, phase_offset, timing_mode, target_fps, beat_state
    )

    # Convert phases to curve values
    beat_vals_1d = torch.tensor(
        [_get_curve_value(curve_name, p) for p in phases],
        device=input_frames_list[0].device,
        dtype=input_frames_list[0].dtype,
    )
    # [T, 1, 1] for broadcasting over C, H, W in [C, T, H, W] tensors
    beat_vals = beat_vals_1d.view(num_frames, 1, 1)

    # Apply effects to each batch item
    result = []
    for frames in input_frames_list:
        modulated = frames.clone()

        if intensity_on and intensity_amt > 0:
            modulated = _apply_intensity(modulated, intensity_amt, beat_vals)

        if blur_on and blur_amt > 0:
            modulated = _apply_blur(modulated, blur_amt, beat_vals_1d)

        if invert_on and invert_amt > 0:
            modulated = _apply_invert(modulated, invert_amt, beat_vals)

        if contrast_on and contrast_amt > 0:
            modulated = _apply_contrast(modulated, contrast_amt, beat_vals)

        modulated = modulated.clamp(-1, 1)
        result.append(modulated)

    return result
