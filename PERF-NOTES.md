# Perf Notes (High Level)

This is a **high-level summary + journey log** of performance work done while building a realtime video pipeline. It is intentionally written without low-level implementation details.

Code map / entrypoints: [HACKATHON-REFERENCE.md](./HACKATHON-REFERENCE.md)

## Goals

- Reduce end-to-end chunk latency and stabilize throughput (avoid periodic stalls).
- Keep output temporally stable across chunk boundaries (cache correctness is as important as raw speed).
- Make performance/debuggability observable (what backend ran, what shapes ran, when caches reset).

## Starting Point → Current

- Starting point: ~11 FPS (early end-to-end baseline with stable output).
- Best observed baseline throughput after core optimizations: ~33 FPS (settings-dependent; after warmup).
- Current “performable” mode: ~23 FPS at 448×448 (B200/B300-class GPUs; includes realtime control/conditioning overhead).

## How We Measured (Practical)

- Measured the system as three rates: **input FPS** (camera/NDI/WebRTC ingest), **pipeline FPS** (generation), and **output pacing FPS** (what viewers actually see).
- Used chunk boundaries as the primary unit of “state commits” (cache resets, parameter application, replay determinism).
- Avoided benchmarking under GPU contention (server still running, another job holding the device), because it makes results noisy and misleading.

## Performance Journey (What Moved the Needle)

### 1) Remove Hidden Caps (Pacing, Contention, Fallbacks)

- Used the measurement split above (input vs pipeline vs pacing) to quickly detect input-limited and output-limited runs.
- Routinely checked for GPU contention (a background server or another job can cut throughput dramatically).
- Made backend selection observable so “silent fallbacks” don’t masquerade as model regressions.

### 2) Make The Hot Path GPU-Efficient

- Integrated a fused attention backend (e.g., FlashAttention 4) where available, with safe fallbacks.
- Focused on the end-to-end critical path: attention + MLP + decode, not just one microkernel.
- Prioritized reducing synchronization points and avoiding accidental host/device round trips.

### 3) Fix Data Movement Before Micro-Optimizing Kernels

- Hunted down implicit copies / contiguity fixes / view-to-contiguous transitions in hot paths (especially decode/resize/resample style code).
- Preferred stable shapes and stable layouts across chunks so caches and compiled graphs can actually be reused.

### 4) Selective Compilation (When It Helps, When It Hurts)

- Used `torch.compile` selectively on stable subgraphs and avoided compile on paths that are shape-volatile or stateful across invocations.
- Accepted that compilation has warmup cost; measured steady-state after warmup.
- Watched for cudagraph / reuse interactions that can surface as “reused output” failures when state persists between calls.

### 5) Cache Hygiene + Transition Semantics (Correctness + Perf)

- Treated chunk boundaries as the primary “state commit” point: cache resets, parameter application, and replay all happen there.
- Made transitions explicit:
  - **Hard cut** = intentional cache reset.
  - **Soft cut** = controlled transition over multiple chunk boundaries.
- Avoided mixing independent encode/decode streams through a shared temporal cache (a common source of boundary artifacts).

### 6) Keep Preprocessing Off The Critical Path

- Depth/control-map generation needs to be fast and predictable, or it becomes the bottleneck (even if generation is fast).
- Prefer asynchronous/pre-buffered preprocessing so occasional slow frames don’t stall the whole pipeline.

### 7) Precision / Quantization Tradeoffs

- Explored mixed precision and (where appropriate) FP8-style quantization to reduce memory bandwidth pressure.
- Kept correctness guardrails so visual quality regressions are obvious and attributable.

## Takeaways

- Most “FPS regressions” weren’t one kernel getting slower — they were fallbacks, extra copies, contention, or a cache/compile mode mismatch.
- Optimizations only stick if they’re observable (backend reporting) and repeatable (benchmark hygiene).
