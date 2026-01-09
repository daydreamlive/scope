# Hackathon Reference (Read-Me-First)

This branch is a **readability-first snapshot** of work added on top of Scope. It is intended for code review / judging and is **not guaranteed to be runnable as-is**.

## Where to Look

- **Realtime control plane (new module)**: `src/scope/realtime/`
  - Event semantics + deterministic chunk-boundary application: `src/scope/realtime/control_bus.py`
  - Prompt sequencing: `src/scope/realtime/prompt_playlist.py`
  - Driver glue: `src/scope/realtime/generator_driver.py`, `src/scope/realtime/pipeline_adapter.py`

- **CLI tools**: `src/scope/cli/`
  - Main CLI entry: `src/scope/cli/video_cli.py`
  - Stream Deck integration: `src/scope/cli/streamdeck_control.py`

- **Server-side recording**: `src/scope/server/session_recorder.py`

- **Input + control-map generation** (depth/edges/composite conditioning): `src/scope/server/frame_processor.py`
  - Vendored depth model used by the control-map pipeline: `src/scope/vendored/video_depth_anything/`

- **VACE integration + chunk-stability work**: `src/scope/core/pipelines/wan2_1/vace/`

- **NDI input support**: `src/scope/server/ndi/`

## What’s Intentionally Not Included

This branch is intentionally scoped to **feature work + readability**. Hardware-specific performance codepaths and low-level optimization infrastructure are out of scope for this public snapshot.

See `PERF-NOTES.md` for a high-level description of performance work (without code).

