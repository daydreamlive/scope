# Samplers

Sampling strategies for the diffusion denoising loop, adapted for flow matching models.

## Available Samplers

| Sampler | Type | Description |
|---------|------|-------------|
| `add_noise` | Stochastic | Original re-noising approach using `scheduler.add_noise()` |
| `euler` | Deterministic | Basic Euler method with optional `s_churn` for stochasticity |
| `euler_ancestral_rf` | Stochastic | Euler Ancestral adapted for Rectified Flow (flow matching) |
| `ipndm` | Multi-step | Adams-Bashforth multi-step (orders 1-4), deterministic |
| `dpmpp_2m` | Multi-step | DPM-Solver++(2M) - second-order using previous denoised |
| `dpmpp_2m_sde` | Multi-step SDE | DPM-Solver++(2M) with stochastic noise (`eta` control) |
| `dpmpp_3m_sde` | Multi-step SDE | DPM-Solver++(3M) - third-order with two history points |
| `gradient_estimation` | Deterministic | Gradient estimation with momentum (`gamma` parameter) |
| `ddpm` | Stochastic | Classic DDPM adapted for sigma parameterization |
| `lcm` | Stochastic | Latent Consistency Model - jumps to denoised + noise |
