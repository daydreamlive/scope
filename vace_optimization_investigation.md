# VACE Optimization Investigation

## Context

Investigated options to reclaim the performance gap between baseline (no VACE) and VACE-enabled generation in the LongLive pipeline at 512x512 with TAE.

## VACE Overhead Breakdown

VACE hints are recomputed **every denoising step** (4 steps/chunk), not once per chunk.

| Component | Per Call | Per Chunk (4 steps) | % of Chunk (~735ms) |
|-----------|---------|-------------------|-------------------|
| VACE forward (15 blocks) | 38.9ms | 155.5ms | 21.2% |
| VAE encode (VACE input) | 13-17ms | 13-17ms (once) | 1.8-2.3% |
| Patch embedding | 0.09ms | 0.36ms | <0.1% |
| **Total VACE overhead** | | **~170ms** | **~23%** |

## Options Evaluated

### 1. torch.compile on VACE Components

**Result: <2% improvement. Not viable.**

- flex_attention is already compiled (`mode="max-autotune-no-cudagraphs"`)
- Linear layers (43.5% of CUDA time) are already cuBLAS-optimized
- SageAttention is a pre-optimized custom CUDA kernel
- TAE is incompatible (MemBlock streaming architecture)

See `vace_torch_compile_report.md` for full benchmark data.

### 2. Cache VACE Hints Across Denoising Steps

**Potential savings: ~117ms/chunk (75% of VACE forward time)**

Compute hints once at step 1, reuse for steps 2-4. However, hints take `x` (the current noisy latent) as input and are designed to adapt to the denoising state at each step. Reusing stale hints means VACE conditioning becomes less precise as denoising progresses.

**Verdict: Likely quality degradation. Not pursued.**

### 3. Reduce VACE Block Count

**Verdict: Destroys quality. Off the table.**

### 4. FP8 Quantization of VACE Blocks

**Potential savings: ~30-35ms/chunk (estimated)**

The pipeline already supports `Float8DynamicActivationFloat8WeightConfig` via torchao for the main model. Could apply same to the 15 VACE blocks. `aten::addmm` is 58.2% of VACE CUDA time — FP8 could roughly halve this.

**Verdict: Viable, but FP8 not yet working on LongLive. Revisit when FP8 support is ready.**

### 5. Fuse VACE Patch Embedding + First Block

**Verdict: Negligible savings. Not worth the effort.**

### 6. CUDA Graphs for VACE Forward

**Potential savings: 25-50ms/chunk (3-7% of total chunk time)**

#### Profiling Data (forward_vace, isolated)

| Metric | Value |
|--------|-------|
| Wall-clock per call | 38.87ms |
| CUDA self time per call | 35.94ms |
| Kernel launches per call | ~922 |
| cudaLaunchKernel CPU overhead per call | ~6.53ms |
| Launch overhead per kernel | ~7.1μs |
| Per chunk (4 steps) total launches | ~3,688 |
| Per chunk cudaLaunchKernel overhead | ~26ms |

#### CUDA Time Breakdown (forward_vace, 5 calls)

| Operation | CUDA Time | % | Calls |
|-----------|-----------|---|-------|
| aten::addmm (linear projections) | 104.6ms | 58.2% | 830 |
| flex_attention (compiled self-attn) | 27.2ms | 15.2% | 75 |
| triton fused kernels (RoPE etc.) | 24.7ms | 13.7% | 75 |
| aten::cat (padding, stacking) | 15.7ms | 8.8% | 760 |
| aten::mul (modulation) | 9.9ms | 5.5% | 1,050 |

#### Estimate

- **Conservative**: ~26-35ms savings/chunk (3.5-4.8%)
- **Optimistic**: ~40-50ms savings/chunk (5.4-6.8%)
- Reclaims roughly **15-30% of the VACE overhead**

#### Blockers (Prerequisites to Implement)

1. **flex_attention compiled with `no-cudagraphs`** — The self-attention kernel explicitly opts out. Would need to either remove this flag or replace with a graph-compatible attention.
2. **`.item()` call in CausalWanSelfAttention** (line 136 of `causal_model.py`) — `s == seq_lens[0].item() * 2` causes GPU→CPU sync. Must be replaced with static branching.
3. **Dynamic control flow** — The `is_tf` branch based on runtime tensor values is incompatible with CUDA graphs.
4. **Growing hint accumulation** — Each VACE block appends hints via `torch.unbind`/`torch.stack`, creating tensors of increasing size across blocks. Requires pre-allocated fixed-size buffer.

#### Verdict

Meaningful but not transformative. 25-50ms/chunk savings requires non-trivial refactoring of the attention path that also affects the non-VACE code. The prerequisites (removing `.item()`, pre-allocating buffers, addressing flex_attention constraint) are significant engineering work for a modest return.

## Summary

| Option | Savings/Chunk | % of Total | Feasibility | Status |
|--------|-------------|-----------|------------|--------|
| torch.compile | ~0ms | <2% | Done | Not viable |
| Hint caching | ~117ms | ~16% | Easy | Quality risk |
| Fewer blocks | Variable | Variable | Easy | Quality risk |
| FP8 VACE blocks | ~30-35ms | ~4-5% | Medium | Blocked (FP8 not ready) |
| Fuse ops | Negligible | <1% | Easy | Not worth it |
| CUDA graphs | 25-50ms | 3-7% | Hard | Significant prerequisites |

## Benchmark Scripts

- `benchmark_vace.py` — Isolated and end-to-end benchmarks with `--compare` mode
- `profile_inpainting.py` — Detailed instrumented profiling
- `profile_torch.py` — Full torch.profiler analysis with chrome trace export
- `profile_vace_overhead.py` — VACE forward kernel launch overhead analysis
