# Ported from https://github.com/thu-ml/TurboDiffusion
# rCM (Rectified Consistency Model) sampling for TurboDiffusion.
#
# Key differences from standard flow matching:
# - Uses TrigFlow parameterization: sigma = sin(t) / (cos(t) + sin(t))
# - Velocity prediction with consistency model stepping
# - Only 1-4 denoising steps needed (vs 50+ for standard diffusion)

import math

import torch


def rcm_sample(
    model,
    context: list[torch.Tensor],
    seq_len: int,
    latent_shape: tuple[int, ...],
    num_steps: int = 4,
    sigma_max: float = 80.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """rCM (Rectified Consistency Model) sampling.

    Generates video latents using timestep-distilled consistency model sampling.
    The model predicts velocity at each step, and the consistency update rule
    produces clean samples in just 1-4 steps.

    Args:
        model: The WanModel (or wrapped model) to use for velocity prediction.
            Called as model(x, t, context, seq_len) where x is a list of
            [C, T, H, W] tensors, t is [B] timesteps, context is list of
            [L, C] text embeddings.
        context: List of text embedding tensors, each [L, C].
        seq_len: Maximum sequence length for positional encoding.
        latent_shape: Shape of latent noise (C, T, H, W) — without batch dim.
        num_steps: Number of sampling steps (1-4).
        sigma_max: Initial sigma for rCM controlling noise magnitude.
        device: Device to generate on.
        dtype: Dtype for model inference.
        generator: Optional torch Generator for reproducible sampling.

    Returns:
        Denoised latent tensor of shape [B, C, T, H, W].
    """
    batch_size = len(context)

    # Generate initial noise
    init_noise = torch.randn(
        batch_size,
        *latent_shape,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    # Build timestep schedule
    # Mid timesteps chosen for visual quality (from TurboDiffusion paper)
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    t_steps = torch.tensor(
        [math.atan(sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Initialize latents scaled by initial timestep
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(batch_size, 1, device=device, dtype=torch.float64)

    # Sampling loop
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        with torch.no_grad():
            # Prepare inputs for WanModel forward:
            # x needs to be list of [C, T, H, W] tensors
            # t needs to be [B] tensor of timesteps (scaled by 1000)
            x_list = [x[i].to(dtype) for i in range(batch_size)]
            timesteps = (t_cur.float() * ones * 1000).squeeze(-1).to(dtype)

            # Model predicts velocity
            v_pred = model(
                x_list,
                timesteps,
                context,
                seq_len,
            )
            v_pred = v_pred.to(torch.float64)

            # Consistency model update step
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )

    return x.to(dtype)
