"""
MagCache utilities for Wan2.1-family diffusion models.

This implements the *runtime* portion of MagCache (NeurIPS 2025):
- Use a pre-calibrated per-step magnitude ratio curve (gamma_t)
- Accumulate an error estimate when skipping consecutive steps
- Reuse a cached residual when the estimated error stays below a threshold

Paper: "MagCache: Fast Video Generation with Magnitude-Aware Cache" (arXiv:2506.09045)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def nearest_interp(src: np.ndarray, target_length: int) -> np.ndarray:
    """Nearest-neighbor interpolate a 1D array to target_length."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    src = np.asarray(src, dtype=np.float64)
    src_length = int(src.shape[0])
    if src_length == 0:
        raise ValueError("src must be non-empty")
    if target_length == 1:
        return np.asarray([src[-1]], dtype=np.float64)
    if src_length == 1:
        return np.repeat(src, target_length).astype(np.float64)

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src[mapped_indices]


# ---------------------------------------------------------------------------
# Pre-calibrated magnitude ratios
# ---------------------------------------------------------------------------
#
# The upstream reference implementation provides "mag_ratios" as an interleaved
# [cond_step0, uncond_step0, cond_step1, uncond_step1, ...] curve and runs the
# model twice per sampling step (CFG). Scope's LongLive pipeline does not do
# CFG in the diffusion wrapper, so we use the *conditional* curve.
#
# Source: https://raw.githubusercontent.com/Zehong-Ma/MagCache/refs/heads/main/MagCache4Wan2.1/magcache_generate.py
#

# Wan2.1 T2V 1.3B, sample_steps=50 (50-step curve), conditional branch only.
# (The upstream array is length 100 with interleaved cond/uncond + initial padding.)
_WAN21_T2V_13B_INTERLEAVED_100 = np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939],
    dtype=np.float64,
)

def wan21_t2v_13b_mag_ratios(num_steps: int) -> np.ndarray:
    """Return conditional-branch mag ratios for Wan2.1 T2V 1.3B."""
    # Conditional ratios are every other value starting at index 0:
    # [cond0, uncond0, cond1, uncond1, ...]
    cond = _WAN21_T2V_13B_INTERLEAVED_100[0::2]
    # The source includes an initial padding entry; keep it (gamma_0 ~= 1.0).
    if cond.shape[0] != 50:
        # Should not happen, but keep robust.
        cond = cond[:50]
    if num_steps == cond.shape[0]:
        return cond.copy()
    return nearest_interp(cond, num_steps)


@dataclass
class MagCacheConfig:
    enabled: bool = False
    # Total accumulated error threshold (δ in the paper)
    thresh: float = 0.12
    # Max consecutive skipped steps (K in the paper)
    # Reference default is 2, which is more conservative for quality
    K: int = 2
    # Fraction of initial steps to preserve unchanged (default 20%)
    retention_ratio: float = 0.2
