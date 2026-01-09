# Perf Notes (High Level)

This is a **high-level summary** of performance work done while building a realtime video pipeline. It is intentionally written without low-level implementation details or kernels.

## Goals

- Reduce end-to-end chunk latency and stabilize throughput (avoid periodic stalls).
- Keep output temporally stable across chunk boundaries (cache correctness is as important as raw speed).
- Make performance/debuggability observable (what backend ran, what shapes ran, when caches reset).

## Themes

### 1) Backend Selection + Visibility

- Integrated a fused attention backend (e.g., FlashAttention 4) where available, with safe fallbacks.
- Built lightweight reporting so it’s obvious which attention/compile backend handled a call, and when fallbacks occur.
- Added “tell me what happened” hooks to catch silent degradations (unexpected graph breaks, backend switching, cache resets).

### 2) Cache Hygiene (Correctness + Perf)

- Treated chunk boundaries as the primary “state commit” point and made cache resets explicit (hard cuts vs seamless continuation).
- Avoided mixing independent encode/decode streams through a shared temporal cache (a common source of boundary artifacts).

### 3) Compile Strategy (When It Helps, When It Hurts)

- Used `torch.compile` selectively on stable subgraphs and avoided compile on paths that are shape-volatile or stateful across invocations.
- Kept an eye on cudagraph interactions and “reused output” style failure modes when state persists between calls.

### 4) Precision / Quantization Tradeoffs

- Explored mixed precision and (where appropriate) FP8-style quantization to reduce memory bandwidth pressure.
- Kept correctness guardrails so visual quality regressions are obvious and attributable.

### 5) Measurement Discipline

- Benchmarked with consistent resolution/settings and avoided GPU contention during measurements.
- Logged enough metadata to reproduce regressions (settings, steps, cache mode, backend mode).
