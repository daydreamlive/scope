# Enhancement techniques for Longlive pipeline
# Based on research from ComfyUI-WanVideoWrapper
#
# References:
# - FreSca: https://github.com/WikiChao/FreSca (MIT License)
# - TSR: https://github.com/temporalscorerescaling/TSR/

import torch
import torch.fft as fft


def fourier_filter(
    x: torch.Tensor,
    scale_low: float | None = 1.0,
    scale_high: float | None = 1.25,
    freq_cutoff: int | None = 20,
) -> torch.Tensor:
    """
    Apply frequency-dependent scaling to enhance details (FreSca).

    This technique scales low and high frequency components differently,
    allowing enhancement of fine details while preserving overall structure.

    Args:
        x: Input tensor of shape [B, F, C, H, W] or [B, C, H, W]
        scale_low: Scaling factor for low-frequency components (structure)
        scale_high: Scaling factor for high-frequency components (details)
        freq_cutoff: Radius in frequency space defining low-frequency region

    Returns:
        Filtered tensor with same shape as input
    """
    # Handle None values with defaults
    if scale_low is None:
        scale_low = 1.0
    if scale_high is None:
        scale_high = 1.25
    if freq_cutoff is None:
        freq_cutoff = 20

    if scale_low == 1.0 and scale_high == 1.0:
        return x

    dtype, device = x.dtype, x.device
    original_shape = x.shape

    # Handle 5D input [B, F, C, H, W] by flattening to 4D
    if x.ndim == 5:
        B, F, C, H, W = x.shape
        x = x.reshape(B * F, C, H, W)
    else:
        H, W = x.shape[-2], x.shape[-1]

    # Convert to float32 for FFT precision
    x = x.to(torch.float32)

    # Apply 2D FFT and shift low frequencies to center
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    # Create frequency mask (real-valued, same shape as x_freq)
    crow, ccol = H // 2, W // 2

    # Clamp freq_cutoff to valid range
    freq_cutoff = min(freq_cutoff, min(crow, ccol))

    # Initialize mask with high-frequency scaling (real tensor, not complex)
    mask = torch.full(x_freq.shape, scale_high, dtype=torch.float32, device=device)

    # Apply low-frequency scaling to center region
    mask[
        ...,
        crow - freq_cutoff : crow + freq_cutoff,
        ccol - freq_cutoff : ccol + freq_cutoff,
    ] = scale_low

    # Apply frequency-specific scaling
    x_freq = x_freq * mask

    # Convert back to spatial domain
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    # Restore original dtype and shape
    x_filtered = x_filtered.to(dtype)
    if len(original_shape) == 5:
        x_filtered = x_filtered.reshape(original_shape)

    return x_filtered


def normalized_fourier_filter(
    x: torch.Tensor,
    scale_low: float | None = 1.0,
    scale_high: float | None = 1.25,
    freq_cutoff: int | None = 20,
    tau: float | None = 1.2,
) -> torch.Tensor:
    """
    Apply frequency-dependent scaling with NAG-style normalization (Normalized FreSca).

    Like fourier_filter, but with self-limiting behavior to prevent accumulation.
    The enhancement is bounded: if the norm ratio exceeds tau, the enhancement
    is scaled back to maintain tau as the maximum amplification.

    This makes the enhancement convergent instead of accumulative:
    - First few chunks see improvement
    - Once norm reaches tau * original, no further boost occurs
    - Prevents runaway amplification over long generations

    Args:
        x: Input tensor of shape [B, F, C, H, W] or [B, C, H, W]
        scale_low: Scaling factor for low-frequency components (structure)
        scale_high: Scaling factor for high-frequency components (details)
        freq_cutoff: Radius in frequency space defining low-frequency region
        tau: Maximum allowed norm ratio (enhancement ceiling). Default 1.2 means
             the enhanced output can be at most 1.2x the original norm.

    Returns:
        Filtered tensor with same shape as input, norm-bounded to tau * original
    """
    # Handle None values with defaults
    if scale_low is None:
        scale_low = 1.0
    if scale_high is None:
        scale_high = 1.25
    if freq_cutoff is None:
        freq_cutoff = 20
    if tau is None:
        tau = 1.2

    if scale_low == 1.0 and scale_high == 1.0:
        return x

    # Apply the base fourier filter
    enhanced = fourier_filter(x, scale_low, scale_high, freq_cutoff)

    # NAG-style normalization to prevent accumulation
    # Compute norms over spatial dimensions (last 2 dims)
    # Keep other dims for broadcasting
    norm_dims = (-2, -1)

    norm_original = torch.norm(x.float(), p=2, dim=norm_dims, keepdim=True)
    norm_enhanced = torch.norm(enhanced.float(), p=2, dim=norm_dims, keepdim=True)

    # Compute ratio (with epsilon for numerical stability)
    ratio = norm_enhanced / (norm_original + 1e-7)

    # Self-limiting: if enhancement exceeds tau, scale it back
    # This prevents accumulation over multiple chunks
    mask = ratio > tau
    adjustment = (norm_original * tau) / (norm_enhanced + 1e-7)

    # Apply adjustment only where ratio exceeds tau
    enhanced = torch.where(mask, enhanced * adjustment.to(enhanced.dtype), enhanced)

    return enhanced


def temporal_score_rescaling(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    timestep: int | torch.Tensor,
    k: float | None = 0.95,
    tsr_sigma: float | None = 0.1,
) -> torch.Tensor:
    """
    Apply temporal score rescaling for more consistent denoising (TSR).

    Adjusts model output based on signal-to-noise ratio at each timestep,
    providing more consistent denoising behavior and improved temporal coherence.

    Args:
        model_output: Model's flow/noise prediction
        sample: Current noisy sample (x_t)
        timestep: Current timestep (0-1000 scale)
        k: Sampling temperature (default 0.95, typical range 0.9-1.0)
        tsr_sigma: SNR influence timing factor (default 0.1)

    Returns:
        Rescaled model output
    """
    # Handle None values with defaults
    if k is None:
        k = 0.95
    if tsr_sigma is None:
        tsr_sigma = 0.1

    # Normalize timestep to [0, 1]
    if isinstance(timestep, torch.Tensor):
        t = (timestep.float().mean() / 1000.0).item()
    else:
        t = timestep / 1000.0

    # Edge cases
    if t == 0.0:
        ratio = k
    elif t == 1.0:
        return model_output
    else:
        # SNR-based ratio calculation
        snr_t = (1 - t) ** 2 / (t**2 + 1e-8)
        ratio = (snr_t * tsr_sigma**2 + 1) / (snr_t * tsr_sigma**2 / k + 1)

    # Apply rescaling
    # model_output = (ratio * ((1-t) * model_output + sample) - sample) / (1 - t)
    rescaled = (ratio * ((1 - t) * model_output + sample) - sample) / (1 - t + 1e-8)

    return rescaled.to(model_output.dtype)


def apply_step_adaptive_scaling(
    x: torch.Tensor,
    step_index: int,
    total_steps: int,
    scale_low: float = 1.0,
    scale_high_start: float = 1.0,
    scale_high_end: float = 1.25,
    freq_cutoff: int = 20,
) -> torch.Tensor:
    """
    Apply FreSca with step-adaptive high-frequency scaling.

    Ramps up detail enhancement through the denoising process:
    - Early steps (high noise): conservative scaling
    - Late steps (low noise): aggressive detail enhancement

    Args:
        x: Input tensor
        step_index: Current step index (0-based)
        total_steps: Total number of denoising steps
        scale_low: Low-frequency scaling (constant)
        scale_high_start: High-frequency scaling at start
        scale_high_end: High-frequency scaling at end
        freq_cutoff: Frequency cutoff radius

    Returns:
        Filtered tensor
    """
    progress = step_index / max(total_steps - 1, 1)
    scale_high = scale_high_start + progress * (scale_high_end - scale_high_start)

    return fourier_filter(x, scale_low, scale_high, freq_cutoff)
