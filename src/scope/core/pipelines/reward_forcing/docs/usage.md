# Reward-Forcing Pipeline

## Overview

Reward-Forcing is a training methodology that enables high-quality video generation in just 4 denoising steps. The distilled model learns from reward signals to produce results comparable to 50+ step diffusion models.

## Key Features

- **Fast Inference**: Only 4 denoising steps required (1000, 750, 500, 250)
- **High Quality**: Maintains visual quality despite fewer steps
- **EMA Sink Mechanism**: Enables real-time streaming with bounded memory
- **Causal Architecture**: Custom causal model optimized for streaming
- **LoRA Support**: Compatible with runtime LoRA adapters

## Architecture

Reward-Forcing uses the Wan2.1-T2V-1.3B architecture with:
- **Causal attention** with EMA sink tokens for bounded memory
- **Local attention** (12 frames) for temporal coherence
- **4-step denoising schedule** for fast generation

### EMA Sink Mechanism

The key innovation is the **Exponential Moving Average (EMA) sink** mechanism:

1. When tokens are evicted from the local attention window (due to cache overflow), they are **compressed** into sink tokens using EMA instead of being discarded
2. Formula: `updated_sink = α * current_sink + (1-α) * evicted_tokens`
3. Default `compression_alpha = 0.999` provides slow, stable updates
4. This enables **bounded memory** while retaining **long-term context**

### How It Works

```
Time →
[Frame 1] [Frame 2] [Frame 3] ... [Frame N]
    ↓ Cache overflow
[Sink Tokens (EMA compressed)] | [Local Window (recent frames)]
```

## Usage

### Basic Usage

```python
from scope.core.pipelines import RewardForcingPipeline

pipeline = RewardForcingPipeline(
    config,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)

# Generate video frames
frames = pipeline(prompt="A panda walking in a park")
```

### Configuration

The pipeline uses these default settings from `model.yaml`:

- `denoising_step_list`: [1000, 750, 500, 250]
- `num_frame_per_block`: 3
- `local_attn_size`: 12
- `sink_size`: 3
- `compression_alpha`: 0.999

### EMA Sink Configuration

```yaml
base_model_kwargs:
  sink_size: 3               # Number of sink frames
  compression_alpha: 0.999   # EMA coefficient (higher = slower update)
```

| Alpha Value | Behavior |
|-------------|----------|
| 0.999 | Very slow update, stable long-term context |
| 0.99 | Moderate update rate |
| 0.9 | Fast update, emphasizes recent context |

## Comparison with LongLive

| Feature | Reward-Forcing | LongLive |
|---------|---------------|----------|
| Denoising Steps | 4 | 4 |
| Sink Mechanism | **EMA compression** | Simple discard |
| Architecture | RewardForcingCausalModel | CausalWanModel |
| Training | Reward-based distillation | Self-Forcing |
| Built-in LoRA | No | Yes (performance adapter) |
| Memory Bound | Yes (via EMA sink) | Yes (via rolling cache) |
| Long-term Context | **Preserved via EMA** | Lost when evicted |

## Reference

Based on: https://github.com/JaydenLu666/Reward-Forcing
