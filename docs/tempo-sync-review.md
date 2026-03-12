# Tempo Sync & Modulation — Implementation Review

Review of the tempo sync and modulation feature as implemented in `marco/feat/beatsync-modulation-2`. Covers architecture, API, data flow, edge cases, and remaining work.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture](#2-architecture)
  - [2.1 Layer 1: Tempo Sources](#21-layer-1-tempo-sources)
  - [2.2 Layer 2: TempoSync Manager](#22-layer-2-temposync-manager)
  - [2.3 Layer 3: ParameterScheduler](#23-layer-3-parameterscheduler)
  - [2.4 Layer 4: ModulationEngine](#24-layer-4-modulationengine)
  - [2.5 Layer 5: Pipeline Injection](#25-layer-5-pipeline-injection)
  - [2.6 Frontend](#26-frontend)
- [3. REST API Reference](#3-rest-api-reference)
- [4. WebRTC Data Channel Messages](#4-webrtc-data-channel-messages)
- [5. Beat State Flow](#5-beat-state-flow)
- [6. Modulation System](#6-modulation-system)
- [7. Demo Pipelines](#7-demo-pipelines)
- [8. Edge Cases and Robustness](#8-edge-cases-and-robustness)
- [9. Feature Gaps and Future Work](#9-feature-gaps-and-future-work)
- [10. File Index](#10-file-index)

---

## 1. Overview

Tempo sync allows Scope's real-time video pipelines to lock to an external beat clock — Ableton Link or MIDI clock — so that visual output reacts to music in time. Target users are VDJs running Scope alongside Ableton Live, Resolume, or similar applications over a shared Link session.

The feature provides three levels of synchronization:

1. **Beat-aware pipelines** — Pipelines receive `bpm`, `beat_phase`, `bar_position`, `beat_count`, and `is_playing` as kwargs on every call.
2. **Parameter scheduling** — Discrete parameter changes (prompt switches, denoising adjustments) can be quantized to beat/bar boundaries with configurable lookahead.
3. **Parameter modulation** — Continuous per-frame modulation of numeric parameters (noise_scale, denoising_steps) driven by beat phase, with configurable wave shapes and rates.

Dependencies are optional: `aalink` for Ableton Link, `mido` + `python-rtmidi` for MIDI clock. When neither is installed, the UI shows install hints and the feature degrades gracefully.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TEMPO SOURCES                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Ableton Link │  │  MIDI Clock  │  │ Client-Forwarded  │ │
│  │  (aalink)    │  │ (mido/rtmidi)│  │ (cloud mode)      │ │
│  │  100Hz poll  │  │  24 PPQN     │  │ via data channel  │ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘ │
└─────────┼─────────────────┼───────────────────┼─────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    TEMPOSYNC MANAGER                         │
│  - Unified BeatState (bpm, phase, bar_pos, count, playing)  │
│  - Client state preferred when fresh (< 2s)                 │
│  - Thread-safe access via locks                             │
│  - 15Hz notification loop → WebRTC data channels            │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌────────────────────────┐    ┌───────────────────────────────┐
│  PARAMETER SCHEDULER   │    │     MODULATION ENGINE         │
│  - Quantize discrete   │    │  - Per-frame param modulation │
│    changes to beat/    │    │  - Wave shapes, rates, depth   │
│    bar boundaries      │    │  - Pipeline-agnostic (kwargs)  │
│  - Lookahead for       │    │  - Pydantic-validated config  │
│    latency comp.       │    │                               │
└────────────────────────┘    └───────────────┬──────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 PIPELINE INJECTION                            │
│  PipelineProcessor injects: beat state + modulated params    │
│  Beat-synced cache reset (optional)                         │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Layer 1: Tempo Sources

**Ableton Link** (`src/scope/server/tempo_sources/link.py`):
- Uses the `aalink` library (async Python wrapper for Ableton Link)
- Joins the shared Link session on the local network automatically
- Polls at ~100Hz via an asyncio task, caches `BeatState` behind a lock
- Reads `link.beat`, `link.tempo`, `link.playing` and derives `beat_phase`, `bar_position`, `beat_count`
- Supports `set_tempo()` to make Scope the session tempo leader
- Exposes `num_peers` for the UI

**MIDI Clock** (`src/scope/server/tempo_sources/midi_clock.py`):
- Uses `mido` with `python-rtmidi` backend
- Runs a dedicated listener thread (daemon) that reads MIDI messages
- Handles: clock (0xF8), start (0xFA), stop (0xFC), continue (0xFB)
- Derives BPM from inter-tick intervals using exponential moving average (alpha=0.15, 24 PPQN)
- Auto-selects the first available MIDI device if none specified

**Client-Forwarded** (cloud mode):
- When running in cloud mode, the browser forwards beat state over the WebRTC data channel
- `TempoSync.update_client_beat_state()` extracts and caches it, strips beat keys from params
- Client state is preferred when fresh (received within the last 2 seconds)

### 2.2 Layer 2: TempoSync Manager

`src/scope/server/tempo_sync.py` is the central hub.

**BeatState** is a frozen dataclass:
```python
@dataclass(frozen=True)
class BeatState:
    bpm: float
    beat_phase: float    # 0.0-1.0 within current beat
    bar_position: float  # 0.0 to beats_per_bar within current bar
    beat_count: int
    is_playing: bool
    timestamp: float
    source: str          # "link", "midi_clock", or "client"
```

**Key behaviors:**
- `get_beat_state()` returns client state if fresh, otherwise server-side source
- `enable()` / `disable()` create and stop the active `TempoSource`
- Notification loop runs at ~15Hz, pushes `tempo_update` to registered WebRTC sessions
- `register_notification_session()` / `unregister_notification_session()` for session lifecycle

### 2.3 Layer 3: ParameterScheduler

`src/scope/server/parameter_scheduler.py` handles beat-quantized discrete parameter changes.

**Quantize modes:** `none` | `beat` | `bar` | `2_bar` | `4_bar`

**Lookahead:** Subtracts `lookahead_ms` from the delay so the visual change lands on the beat despite pipeline latency.

**Merging:** If a timer is already pending, new params merge into the existing schedule (last-write-wins). The original target boundary is preserved.

**Test coverage:** `tests/test_parameter_scheduler.py` — adversarial tests for boundary math, concurrency, lookahead, and edge cases.

### 2.4 Layer 4: ModulationEngine

`src/scope/server/modulation.py` — pipeline-agnostic parameter modulation.

**Design:** Operates on the kwargs dict *before* it reaches any pipeline. No diffusers-specific coupling. Config is validated with Pydantic (`ModulationConfig`).

**Wave shapes:** `sine`, `cosine`, `triangle`, `saw`, `square`, `exp_decay`

**Rates:** `half_beat`, `beat`, `2_beat`, `bar`, `2_bar`, `4_bar`

**Config per target:**
- `enabled`, `shape`, `depth` (0–1), `rate`
- `base_value`, `min_value`, `max_value` (optional bounds)

**Targets:** Derived from pipeline schema fields with `ui.modulatable === true`. Currently: `noise_scale`, `denoising_steps` (aliased to `denoising_step_list`) on LongLive, Krea, MemFlow, RewardForcing, StreamDiffusionV2.

**Side effects:**
- When modulating `noise_scale`: sets `noise_controller=False` so motion-aware controller doesn't overwrite
- When modulating `denoising_step_list`: sets `_modulated_step_list=True` so `SetTimestepsBlock` skips cache reset every frame

**Param alias:** `denoising_steps` → `denoising_step_list`

### 2.5 Layer 5: Pipeline Injection

`PipelineProcessor` injects beat state and applies modulation on every pipeline call:

1. Get `beat_state` from `tempo_sync.get_beat_state()`
2. Inject `bpm`, `beat_phase`, `bar_position`, `beat_count`, `is_playing` into `call_params`
3. Call `modulation_engine.apply(beat_state, call_params)` to modulate enabled targets
4. Optionally trigger beat-synced cache reset (`_beat_cache_reset_rate`: `beat`, `bar`, `2_bar`, `4_bar`) with new `base_seed`

### 2.6 Frontend

**`useTempoSync`** (`frontend/src/hooks/useTempoSync.ts`):
- Manages tempo state, sources, enable/disable, set BPM
- Polls `/api/v1/tempo/status` when enabled; `updateFromNotification()` for data channel updates

**`TempoSyncPanel`** / **`TempoSyncSection`**:
- Toggle tempo sync, source selection (Link / MIDI), MIDI device picker
- Beats per bar, BPM input (Link only), live beat indicator
- Quantize mode, lookahead slider
- Beat cache reset rate (none / beat / bar / 2 bars / 4 bars)
- Prompt cycle rate (cycles through timeline prompts on boundaries)
- **ModulationSection** (nested)

**`ModulationSection`** (`frontend/src/components/settings/ModulationSection.tsx`):
- Target selector (from schema `ui.modulatable` fields)
- Shape, rate, depth, base value, min/max per target
- Sends `modulations` dict via parameter updates (WebRTC data channel)

**Prompt cycling:** Implemented in `StreamPage` — when `promptCycleRate` is set, advances through timeline prompts on each beat/bar boundary. Uses `beatCount` from tempo state; sends discrete `prompts` updates.

---

## 3. REST API Reference

All tempo endpoints are under `/api/v1/tempo/`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/tempo/status` | Current tempo status and beat state |
| POST | `/api/v1/tempo/enable` | Enable tempo sync (Link or MIDI) |
| POST | `/api/v1/tempo/disable` | Disable tempo sync |
| POST | `/api/v1/tempo/set_tempo` | Set BPM (Link only) |
| GET | `/api/v1/tempo/sources` | Available tempo sources and capabilities |

**Modulation:** No dedicated REST API. Config is sent as `modulations` in WebRTC parameter updates; `FrameProcessor` routes it to `ModulationEngine.update()`.

---

## 4. WebRTC Data Channel Messages

### Server → Client

**`tempo_update`** (pushed at ~15Hz when tempo sync is enabled):
```json
{
  "type": "tempo_update",
  "bpm": 128.0,
  "beat_phase": 0.35,
  "bar_position": 2.35,
  "beat_count": 142,
  "is_playing": true
}
```

**`change_scheduled`** / **`change_applied`** — from ParameterScheduler when quantized updates are queued and applied.

### Client → Server

Parameter updates can include:
- `quantize_mode`, `lookahead_ms` — routed to ParameterScheduler
- `modulations` — routed to ModulationEngine
- `beat_cache_reset_rate` — routed to PipelineProcessors
- `bpm`, `beat_phase`, `bar_position`, `beat_count`, `is_playing` — routed to TempoSync (cloud mode)

---

## 5. Beat State Flow

### Local Mode

```
Ableton Link / MIDI device → TempoSource → TempoSync.get_beat_state()
    → PipelineProcessor (inject + modulate) → Pipeline
    → _notification_loop() → WebRTC data channel → useTempoSync
```

### Cloud Mode

```
Browser (Link client) → WebRTC param message (beat keys)
    → FrameProcessor.update_parameters() → TempoSync.update_client_beat_state()
    → PipelineProcessor (same injection as local)
```

---

## 6. Modulation System

**Data flow:**
```
ModulationSection (UI) → modulations state → sendParameterUpdate({ modulations })
    → WebRTC data channel → FrameProcessor.update_parameters()
    → ModulationEngine.update(raw_configs)
    → (per-frame) ModulationEngine.apply(beat_state, call_params)
    → PipelineProcessor passes modulated params to pipeline
```

**ModulationConfig** (Pydantic):
```python
enabled: bool = True
shape: WaveShape = "cosine"   # sine, cosine, triangle, saw, square, exp_decay
depth: float = 0.3            # 0–1
rate: ModulationRate = "bar"  # half_beat, beat, 2_beat, bar, 2_bar, 4_bar
base_value: float | None = None
min_value: float | None = None
max_value: float | None = None
```

**Modulatable pipelines:** LongLive, Krea, MemFlow, RewardForcing, StreamDiffusionV2 expose `denoising_steps` and (LongLive) `noise_scale` as modulatable via `ui_field_config(modulatable=True, modulatable_min=..., modulatable_max=...)`.

---

## 7. Demo Pipelines

### metronome

Visual metronome for testing ParameterScheduler lookahead:
- Beat pulse, beat/bar numbers, BPM display
- Three color layers (A/B/C) that mix additively
- `latency_ms` parameter simulates pipeline processing delay
- Workflow: set latency, toggle layers with quantize enabled, adjust lookahead until changes land on the beat

### mod-scope

Oscilloscope for modulation visualization:
- Renders traces for `noise_scale`, `vace_context_scale`, `kv_cache_attention_bias`
- Use to verify ModulationEngine output in real time

---

## 8. Edge Cases and Robustness

### Strengths

- **Thread safety:** Separate locks for source access, client state, notification sessions. BeatState is immutable.
- **Graceful degradation:** Works without `aalink`/`mido`; UI shows install hints.
- **Cloud mode:** Client-forwarded beat state for deployments without Link/MIDI.
- **Pipeline-agnostic modulation:** ModulationEngine operates on kwargs; any pipeline accepting the params gets modulation.
- **Pydantic validation:** Modulation config validated on receipt.
- **Test coverage:** ParameterScheduler has adversarial tests.

### Known Limitations

| Issue | Severity | Notes |
|-------|----------|-------|
| `threading.Timer` not monotonic | Low | NTP corrections rarely matter for sub-second timing |
| Link reconnection silent | Low | No UI/log when peers drop to 0 |
| MIDI clock jitter | Low | EMA smoothing; unstable interfaces may jitter |
| Client staleness 2s hardcoded | Low | Not configurable |
| No `is_playing` in ParameterScheduler | Medium | Transport stopped; scheduler still computes delays |
| No scheduler flush on disable | Medium | Pending timer fires after tempo sync disabled |
| Notification loop swallows errors | Low | Failed sessions stay registered |

---

## 9. Feature Gaps and Future Work

### Near-term

| Feature | Description |
|---------|-------------|
| Tap tempo | Frontend BPM source for users without Link/MIDI |
| `beats_per_bar` in pipelines | Pass as kwarg for correct time signatures |
| `is_playing` in ParameterScheduler | Apply immediately when transport stopped |
| `cancel_pending()` on disable | Flush scheduler when tempo sync disabled |
| Tempo API docs | OpenAPI/Swagger for REST endpoints |

### Medium-term

| Feature | Description |
|---------|-------------|
| OSC input | Accept BPM/beat from OSC messages |
| MIDI note/CC mapping | Map notes/CC to parameter changes |
| Modulation presets | Save/load effect configs |
| Multi-rate modulation | Different effects at different divisions |

### Long-term

| Feature | Description |
|---------|-------------|
| Resolume Wire protocol | Direct parameter integration |
| Ableton Max for Live device | Control Scope from Ableton |
| Multi-session sync | Multiple Scope instances sharing beat state |

---

## 10. File Index

### Backend

| File | Role |
|------|------|
| `src/scope/server/tempo_sync.py` | TempoSync, BeatState, TempoSource ABC |
| `src/scope/server/parameter_scheduler.py` | Beat-quantized parameter scheduling |
| `src/scope/server/modulation.py` | ModulationEngine, ModulationConfig |
| `src/scope/server/tempo_sources/link.py` | Ableton Link adapter |
| `src/scope/server/tempo_sources/midi_clock.py` | MIDI clock adapter |
| `src/scope/server/frame_processor.py` | TempoSync, ParameterScheduler, ModulationEngine integration |
| `src/scope/server/pipeline_processor.py` | Beat injection, modulation apply, beat cache reset |
| `src/scope/server/webrtc.py` | Session registration for tempo notifications |
| `src/scope/server/app.py` | Tempo REST endpoints |
| `src/scope/server/schema.py` | Pydantic models for tempo API |

### Frontend

| File | Role |
|------|------|
| `frontend/src/hooks/useTempoSync.ts` | Tempo state management |
| `frontend/src/components/TempoSyncPanel.tsx` | Card wrapper for tempo UI |
| `frontend/src/components/settings/TempoSyncSection.tsx` | Tempo controls, quantize, modulation |
| `frontend/src/components/settings/ModulationSection.tsx` | Modulation target/shape/rate UI |
| `frontend/src/lib/api.ts` | Tempo API types and functions |
| `frontend/src/hooks/useUnifiedWebRTC.ts` | Data channel tempo message handling |
| `frontend/src/pages/StreamPage.tsx` | Quantize, prompt cycling, tempo wiring |

### Pipelines

| File | Role |
|------|------|
| `src/scope/core/pipelines/metronome/` | Visual metronome for lookahead testing |
| `src/scope/core/pipelines/mod_scope/` | Modulation oscilloscope |
| `src/scope/core/pipelines/base_schema.py` | `ui_field_config(modulatable=...)` |
| `src/scope/core/pipelines/longlive/schema.py` | modulatable: noise_scale, denoising_steps |
| `src/scope/core/pipelines/krea_realtime_video/schema.py` | modulatable: denoising_steps |
| `src/scope/core/pipelines/memflow/schema.py` | modulatable: denoising_steps |
| `src/scope/core/pipelines/reward_forcing/schema.py` | modulatable: denoising_steps |
| `src/scope/core/pipelines/streamdiffusionv2/schema.py` | modulatable: denoising_steps |

### Tests

| File | Role |
|------|------|
| `tests/test_parameter_scheduler.py` | ParameterScheduler adversarial tests |

### Dependencies

```toml
[project.optional-dependencies]
link = ["aalink>=0.1.1"]
midi = ["mido>=1.3.0", "python-rtmidi>=1.5.0"]
```
