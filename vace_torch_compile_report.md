# VACE torch.compile Performance Report

## Overview

Investigated torch.compile optimization potential for VACE components in the LongLive pipeline, with specific focus on continual conditioning (depth maps) and inpainting tasks at 512x512 resolution using TAE.

## Test Setup

- **GPU**: NVIDIA RTX (Windows 11)
- **Model**: LongLive-1.3B with VACE 1.3B
- **VAE**: TAE (Tiny Autoencoder)
- **Resolution**: 512x512
- **Chunk size**: 12 frames per chunk (3 latent frames)
- **Denoising steps**: 4

## Benchmark Results

### Isolated Component Timings (Baseline)

| Component | Time (ms) | % of Chunk |
|-----------|-----------|------------|
| VAE Encode (single stream) | 12.6 | 1.7% |
| VAE Encode (dual stream, inpainting) | 17.2 | 2.3% |
| VACE Patch Embedding | 0.09 | <0.01% |
| VACE Forward (hint generation, 15 blocks) | 38.9 | 5.3% |
| VAE Decode | 13.8 | 1.9% |
| **Total VACE overhead** | **~52** | **~7%** |
| **Full chunk (e2e)** | **~735** | **100%** |

### torch.compile Results (VACE patch embedding + 15 VACE blocks)

| Metric | Baseline | Compiled | Change |
|--------|----------|----------|--------|
| Depth per chunk | 739ms | 742ms | -0.5% (no change) |
| Depth FPS | 16.25 | 16.17 | -0.5% |
| Inpainting per chunk | 745ms | 735ms | +1.3% |
| Inpainting FPS | 16.11 | 16.33 | +1.4% |
| VACE forward (isolated) | 38.9ms | 38.1ms | +2.0% |
| VACE patch emb (isolated) | 0.09ms | 0.08ms | ~0% |

**Result: torch.compile provides negligible speedup (<2%) on VACE components.**

### Why torch.compile Doesn't Help

1. **flex_attention is already compiled** - The most expensive operation in VACE blocks (self-attention) uses `torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")` which is already optimized.
2. **Linear layers dominate** - `aten::addmm` (43.5% of CUDA time) is already handled by cuBLAS with near-optimal performance for these tensor sizes.
3. **SageAttention is a custom kernel** - The cross-attention uses SageAttention (62ms, 9.8%) which is a pre-optimized CUDA kernel that torch.compile cannot improve.
4. **TAE is incompatible** - TAE's MemBlock architecture with streaming state prevents torch.compile from wrapping the encoder/decoder (requires special `past` argument handling during iteration).

### Depth Conditioning vs Inpainting

| Metric | Depth | Inpainting | Overhead |
|--------|-------|------------|----------|
| E2E per chunk | 730ms | 736ms | +6ms (+0.8%) |
| VAE encode | 8.7ms | 9.0ms | +0.3ms |

**Inpainting adds <1% overhead compared to depth conditioning.** The dual-stream VAE encoding (inactive + reactive) for inpainting costs only ~0.3-5ms extra. The "huge performance hit" for inpainting is NOT in the VACE encoding or hint generation path.

## Detailed Profiling: Where Time Actually Goes

Using `torch.profiler`, the breakdown for a single inpainting chunk (735ms e2e):

### CUDA Time Breakdown (631ms total GPU time)

| Operation | CUDA Time | % | Calls | Description |
|-----------|-----------|---|-------|-------------|
| aten::addmm | 274ms | 43.5% | 1,894 | Linear projections (Q/K/V, FFN, etc.) |
| aten::mm | 81ms | 12.8% | 2,400 | Matrix multiplies (RoPE, modulation) |
| sageattn | 62ms | 9.8% | 150 | SageAttention cross-attention |
| aten::copy_ | 40ms | 6.3% | 3,361 | Tensor copies (KV cache updates) |
| aten::mul | 35ms | 5.5% | 4,240 | Element-wise multiplies |
| aten::cat | 32ms | 5.0% | 1,888 | Tensor concatenation (cache assembly) |
| Memcpy DtoD | 30ms | 4.7% | 1,830 | Device-to-device copies (cache rolling) |
| Convolution | 24ms | 3.8% | 114 | VAE + patch embedding |
| flex_attention | 20ms | 3.2% | 60 | Compiled self-attention |

### CPU Bottlenecks

| Operation | CPU Time | Calls | Issue |
|-----------|----------|-------|-------|
| cudaLaunchKernel | 104ms | 17,020 | Kernel launch overhead |
| aten::item() | 51ms | 846 | GPU→CPU sync (scalar reads) |
| cudaStreamSynchronize | 43ms | 631 | Stream synchronization |
| cudaMemcpyAsync | 21ms | 2,461 | Memory transfers |

## Root Cause Analysis: Inpainting Performance

The true bottlenecks are NOT VACE-specific:

### 1. KV Cache Management (est. ~100ms overhead)
- **3,361 copy operations** (40ms) and **1,830 memcpy D2D** (30ms) from cache rolling
- **1,888 cat operations** (32ms) from cache assembly in `CausalWanSelfAttention`
- The `_apply_cache_updates` method clones and rolls KV cache tensors per-block per-step

### 2. Kernel Launch Overhead (~104ms)
- 17,020 kernel launches averaging 6μs each on CPU
- The modular pipeline framework adds dispatch overhead per block

### 3. GPU-CPU Synchronization (~51ms)
- 846 `aten::item()` calls force GPU→CPU sync
- Likely from KV cache index management (`kv_cache["global_end_index"].item()`, `kv_cache["local_end_index"].item()`)

### 4. Tensor Dtype Conversion (~22ms)
- 2,261 `_to_copy` operations for dtype conversion between bf16/fp32

## Recommendations

### What Won't Help
- **torch.compile on VACE components** - Already <2% impact, flex_attention already compiled
- **torch.compile on TAE** - Incompatible with MemBlock streaming architecture

### What Could Help (Future Work)

1. **Eliminate `aten::item()` calls in KV cache** - Replace scalar reads with tensor operations to avoid GPU→CPU sync. The 846 `.item()` calls in `CausalWanSelfAttention` cost 51ms.

2. **Pre-allocate and reuse KV cache tensors** - Instead of `.clone()` + rolling on every step, use circular buffer indexing. This would reduce the 3,361 copy operations and 1,830 memcpy calls.

3. **Batch cache updates** - The current per-block cache update pattern could be batched to reduce kernel launches.

4. **CUDA graphs for the denoising loop** - Since input shapes are fixed per-chunk, CUDA graphs could eliminate the 17K kernel launch overhead. This requires careful handling of the KV cache state.

5. **Use WanVAE instead of TAE for compile benefits** - WanVAE's Encoder3d/Decoder3d can be directly compiled (no MemBlock issue). However, WanVAE is slower by default, so the net benefit depends on compile speedup.

## Benchmark Scripts

- `benchmark_vace.py` - Isolated and end-to-end benchmarks with `--compare` mode
- `profile_inpainting.py` - Detailed instrumented profiling
- `profile_torch.py` - Full torch.profiler analysis with chrome trace export

### Usage
```bash
# Compare baseline vs torch.compile
python -m scope.core.pipelines.longlive.benchmark_vace --compare

# Isolated component benchmarks only
python -m scope.core.pipelines.longlive.benchmark_vace --mode isolated

# Detailed profiling
python -m scope.core.pipelines.longlive.profile_inpainting

# Full torch profiler with trace export
python -m scope.core.pipelines.longlive.profile_torch
```
