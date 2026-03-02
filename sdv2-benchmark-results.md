# StreamDiffusionV2 Benchmark Results

GPU: NVIDIA GeForce RTX 5090, Windows 11
Config: 2-step denoising ([700, 500, 0]), video-to-video mode
Video: original.mp4 from StreamDiffusionV2 repo (81 frames)

## Original Repo (CausalStreamInferencePipeline)

Component-level breakdown:

| Resolution | VAE Encode | DiT | VAE Decode | Total | FPS |
|-----------|-----------|-----|-----------|-------|-----|
| 208x208 | 11ms (8%) | 108ms (78%) | 20ms (14%) | 139ms | 28.8 |
| 512x512 | 66ms (20%) | 147ms (46%) | 110ms (34%) | 323ms | 12.4 |
| 480x832 | 100ms (23%) | 168ms (39%) | 168ms (39%) | 435ms | 9.2 |

## Scope (StreamDiffusionV2Pipeline, no plugins)

| Resolution | Avg Latency | FPS |
|-----------|------------|-----|
| 208x208 | 164ms | 24.3 |
| 512x512 | 313ms | 12.8 |
| 480x832 | 427ms | 9.4 |

## Comparison

| Resolution | Original Repo | Scope | Difference |
|-----------|--------------|-------|-----------|
| 208x208 | 28.8 fps | 24.3 fps | -16% |
| 512x512 | 12.4 fps | 12.8 fps | +3% |
| 480x832 | 9.2 fps | 9.4 fps | +2% |

## Plugin Chain (PipelineProcessor, 208x208)

Tested with audio-transcription plugin chained as preprocessor to SDv2, using
Scope's production PipelineProcessor threading model. 30fps input, 30s duration.

| Configuration | Throughput FPS | Frames Out |
|--------------|---------------|------------|
| SDv2 only (PipelineProcessor) | 24.5 | 721 |
| With plugin (time.sleep removed) | 25.4 | 1013 |
| With plugin (time.sleep(1.0) present) | ~0.8 | ~24 (estimated) |

The plugin's `time.sleep(1.0)` in `__call__` is the entire bottleneck. With sleep
removed, the plugin chain runs at full speed, essentially identical to SDv2 alone.

## Root Cause of Issue #559

The issue reporter observed low FPS and added `time.sleep(1.0)` to the plugin's
`__call__` method to throttle the preprocessor. This sleep caps the preprocessor at
1 frame/sec. Since SDv2 needs 4 frames per chunk, it must wait ~4 seconds to
accumulate enough input, resulting in the observed 0.5-1 fps.

Without the sleep, the plugin is transparent (negligible overhead), and the chain
runs at 24-25 fps at 208x208.

## Notes

- Original repo's paper claims (4x H100, multi-GPU pipeline parallel): 64.52 fps (1.3B, 1-step)
- Single-GPU numbers are not published; their multi-GPU demo hides VAE behind pipeline parallelism
- Scope uses SageAttention + flash-attn 2; original repo uses flash-attn 2 only
- At 512x512 and 480x832, Scope matches or slightly beats the original repo
- At 208x208, Scope has ~16% overhead (likely preprocessing/postprocessing at small sizes)
- VRAM usage: 9.8 GB (208x208), 12.3 GB (512x512), 13.9 GB (480x832) in Scope
