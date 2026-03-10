# Tempo Sync Feature Review

Review of the `marco/feat/beatsync` branch and the `marco/feat/beatsync-modulation` branch, covering architecture, API, edge cases, modulation analysis, and recommendations for merging to main.

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture](#2-architecture)
  - [2.1 Layer 1: Tempo Sources](#21-layer-1-tempo-sources)
  - [2.2 Layer 2: TempoSync Manager](#22-layer-2-temposync-manager)
  - [2.3 Layer 3: ParameterScheduler](#23-layer-3-parameterscheduler)
  - [2.4 Layer 4: Pipeline Injection](#24-layer-4-pipeline-injection)
  - [2.5 Frontend](#25-frontend)
- [3. REST API Reference](#3-rest-api-reference)
- [4. WebRTC Data Channel Messages](#4-webrtc-data-channel-messages)
- [5. Beat State Flow](#5-beat-state-flow)
  - [5.1 Local Mode](#51-local-mode)
  - [5.2 Cloud Mode](#52-cloud-mode)
- [6. Parameter Scheduling](#6-parameter-scheduling)
- [7. Demo Pipelines](#7-demo-pipelines)
- [8. Edge Cases and Robustness Analysis](#8-edge-cases-and-robustness-analysis)
- [9. Modulation Branch Analysis](#9-modulation-branch-analysis)
  - [9.1 What it adds](#91-what-it-adds)
  - [9.2 What it deletes](#92-what-it-deletes)
  - [9.3 Problems](#93-problems)
  - [9.4 What to salvage](#94-what-to-salvage)
- [10. Recommendations for Main](#10-recommendations-for-main)
  - [10.1 Merge the beatsync branch as-is](#101-merge-the-beatsync-branch-as-is)
  - [10.2 Redesign modulation as a pipeline-agnostic system](#102-redesign-modulation-as-a-pipeline-agnostic-system)
  - [10.3 Proposed modulation architecture](#103-proposed-modulation-architecture)
- [11. Feature Gaps and Future Work](#11-feature-gaps-and-future-work)
- [12. File Index](#12-file-index)

---

## 1. Overview

Tempo sync allows Scope's real-time video pipelines to lock to an external beat clock -- Ableton Link or MIDI clock -- so that visual output reacts to music in time. The target users are VDJs running Scope alongside Ableton Live, Resolume, or similar applications over a shared Link session.

The feature provides two levels of synchronization:

1. **Beat-aware pipelines** -- Pipelines receive `bpm`, `beat_phase`, `bar_position`, `beat_count`, and `is_playing` as kwargs on every call. They can use this to drive any beat-reactive rendering.
2. **Parameter scheduling** -- Discrete parameter changes (prompt switches, denoising adjustments, etc.) can be quantized to beat/bar boundaries with configurable lookahead to compensate for pipeline processing latency.

Dependencies are optional: `aalink` for Ableton Link, `mido` + `python-rtmidi` for MIDI clock. When neither is installed, the UI shows install hints and the feature degrades gracefully.

---

## 2. Architecture

The system is layered into four concerns: tempo sources, the central manager, parameter scheduling, and pipeline injection.

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
│  PARAMETER SCHEDULER   │    │     PIPELINE INJECTION        │
│  - Quantize to beat/   │    │  PipelineProcessor injects:   │
│    bar/2-bar/4-bar     │    │    bpm, beat_phase,           │
│  - Lookahead for       │    │    bar_position, beat_count,  │
│    latency comp.       │    │    is_playing                 │
│  - Merge pending       │    │  into every pipeline call     │
│    params into one     │    │                               │
│    scheduled apply     │    │                               │
└────────────────────────┘    └───────────────────────────────┘
```

### 2.1 Layer 1: Tempo Sources

**Ableton Link** (`src/scope/server/tempo_sources/link.py`):
- Uses the `aalink` library (async Python wrapper for Ableton Link)
- Joins the shared Link session on the local network automatically
- Polls at 100Hz via an asyncio task, caches `BeatState` behind a lock
- Reads `link.beat`, `link.tempo`, `link.playing` and derives `beat_phase` (beat % 1.0), `bar_position` (beat % beats_per_bar), `beat_count` (int(beat))
- Supports `set_tempo()` to make Scope the session tempo leader
- Exposes `num_peers` for the UI to display Link peer count

**MIDI Clock** (`src/scope/server/tempo_sources/midi_clock.py`):
- Uses `mido` with `python-rtmidi` backend
- Runs a dedicated listener thread (daemon) that reads MIDI messages
- Handles: clock (0xF8) for tick counting, start (0xFA) for reset, stop (0xFC), continue (0xFB)
- Derives BPM from inter-tick intervals using exponential moving average (alpha=0.15, 24 PPQN)
- Auto-selects the first available MIDI device if none specified

**Client-Forwarded** (cloud mode):
- When running in cloud mode, the browser client forwards beat state parameters (`bpm`, `beat_phase`, `bar_position`, `beat_count`, `is_playing`) over the WebRTC data channel
- `TempoSync.update_client_beat_state()` extracts these from the parameter dict, caches them as a `BeatState` with `source="client"`, and strips them from the params to avoid double-injection
- Client state is preferred over server-side sources when fresh (received within the last 2 seconds)

### 2.2 Layer 2: TempoSync Manager

`src/scope/server/tempo_sync.py` is the central hub.

**BeatState** is a frozen dataclass:
```python
@dataclass(frozen=True)
class BeatState:
    bpm: float           # Current tempo
    beat_phase: float    # 0.0-1.0 within current beat
    bar_position: float  # 0.0 to beats_per_bar within current bar
    beat_count: int      # Monotonically increasing beat counter
    is_playing: bool     # Transport state
    timestamp: float     # time.time() when captured
    source: str          # "link", "midi_clock", or "client"
```

**Key behaviors:**
- `get_beat_state()` returns client state if fresh, otherwise falls back to the server-side source. This is thread-safe via separate locks for client state and source access.
- `enable()` creates and starts the appropriate `TempoSource`, then starts the notification loop.
- `disable()` stops the notification loop and source.
- The notification loop runs at ~15Hz and pushes `tempo_update` messages to all registered WebRTC sessions via their data channels.
- Session registration/unregistration is handled via `register_notification_session()` / `unregister_notification_session()`.

### 2.3 Layer 3: ParameterScheduler

`src/scope/server/parameter_scheduler.py` handles beat-quantized parameter changes.

The core idea: instead of applying parameter changes immediately (which would land at random points in the beat), the scheduler computes the wall-clock time of the next beat/bar boundary, subtracts a user-configured lookahead (compensating for pipeline latency), and fires a `threading.Timer` to apply the parameters at that moment.

**Quantize modes:**
| Mode | Boundary |
|------|----------|
| `none` | Immediate (no scheduling) |
| `beat` | Next beat |
| `bar` | Next bar downbeat |
| `2_bar` | Next 2-bar boundary |
| `4_bar` | Next 4-bar boundary |

**Lookahead:** Subtracts `lookahead_ms` from the delay so the visual change lands on the beat despite pipeline processing time. If lookahead exceeds the time to the next boundary, the scheduler targets a later cycle boundary (never goes negative).

**Merging:** If a timer is already pending, new params merge into the existing schedule without recomputing the target. This means rapid parameter changes accumulate into a single apply at the original target time.

### 2.4 Layer 4: Pipeline Injection

`PipelineProcessor` (lines 450-457 of `pipeline_processor.py`) injects beat state into every pipeline call:

```python
if self.tempo_sync is not None:
    beat_state = self.tempo_sync.get_beat_state()
    if beat_state is not None:
        call_params["bpm"] = beat_state.bpm
        call_params["beat_phase"] = beat_state.beat_phase
        call_params["bar_position"] = beat_state.bar_position
        call_params["beat_count"] = beat_state.beat_count
        call_params["is_playing"] = beat_state.is_playing
```

This means any pipeline automatically receives beat state without needing to know about TempoSync. Pipelines that don't use these kwargs simply ignore them.

`FrameProcessor` sits above `PipelineProcessor` and:
- Holds the `TempoSync` instance
- Creates a `ParameterScheduler` (if tempo_sync is provided) with `update_parameters` as the apply callback
- Intercepts `quantize_mode` and `lookahead_ms` from incoming parameters and routes them to the scheduler
- Calls `tempo_sync.update_client_beat_state()` on every parameter update to handle cloud-forwarded beat state

### 2.5 Frontend

**`useTempoSync` hook** (`frontend/src/hooks/useTempoSync.ts`):
- Manages `TempoState` (enabled, bpm, beatPhase, barPosition, beatCount, isPlaying, sourceName, numPeers, beatsPerBar)
- Provides `enable()`, `disable()`, `setSessionTempo()`, `fetchSources()` actions
- `updateFromNotification()` applies real-time updates from the WebRTC data channel
- Polls `/api/v1/tempo/status` every 2s as fallback when data channel isn't available

**`TempoSyncSection` component** (`frontend/src/components/settings/TempoSyncSection.tsx`):
- Toggle to enable/disable tempo sync
- Source selection (Ableton Link / MIDI Clock) with availability detection
- MIDI device picker with refresh button
- Beats per bar selector (2-8)
- Live BPM display with beat indicator (pulsing dot)
- Set BPM control (Link only, 20-300 range)
- Beat Quantize mode dropdown (Immediate / Beat / Bar / 2 Bars / 4 Bars)
- Lookahead slider (0-1000ms, shown when quantize is not "none")
- Link peer count display
- Install hints when no sources are available

**WebRTC integration** (`useUnifiedWebRTC.ts`):
- Handles `tempo_update` messages from the data channel
- Handles `change_scheduled` and `change_applied` notifications from the ParameterScheduler
- Sends `quantize_mode` and `lookahead_ms` as regular parameters
- When quantize is active, `StreamPage` flags discrete params with `_quantized: true` and routes them through `schedule_quantized_update`

---

## 3. REST API Reference

All endpoints are under `/api/v1/tempo/`.

### `GET /api/v1/tempo/status`

Returns current tempo sync status including live beat state.

**Response** (`TempoStatusResponse`):
```json
{
  "enabled": true,
  "source": { "type": "link", "num_peers": 2 },
  "beats_per_bar": 4,
  "beat_state": {
    "bpm": 128.0,
    "beat_phase": 0.35,
    "bar_position": 2.35,
    "beat_count": 142,
    "is_playing": true,
    "source": "link"
  }
}
```

### `POST /api/v1/tempo/enable`

Enable tempo sync with a source.

**Request** (`TempoEnableRequest`):
```json
{
  "source": "link",        // "link" | "midi_clock"
  "bpm": 120.0,            // Initial BPM (Link only), 20-300
  "beats_per_bar": 4,      // 1-16
  "midi_device": null       // Required when source is "midi_clock"
}
```

**Response:** `TempoStatusResponse` (same as status)

**Errors:**
- 400 if dependency not installed
- 400 if MIDI device not found
- 500 if TempoSync not initialized

### `POST /api/v1/tempo/disable`

Disable tempo sync and stop the active source.

**Response:** `TempoStatusResponse` with `enabled: false`

### `POST /api/v1/tempo/set_tempo`

Set the session BPM. Only works with Ableton Link (makes Scope the tempo leader).

**Request** (`TempoSetTempoRequest`):
```json
{ "bpm": 140.0 }
```

**Response:** `TempoStatusResponse` with updated BPM

**Errors:**
- 400 if tempo sync not enabled
- 400 if source doesn't support set_tempo (MIDI clock)

### `GET /api/v1/tempo/sources`

List available tempo sources and their capabilities.

**Response** (`TempoSourcesResponse`):
```json
{
  "sources": {
    "link": {
      "available": true,
      "name": "Ableton Link"
    },
    "midi_clock": {
      "available": true,
      "name": "MIDI Clock",
      "devices": ["IAC Driver Bus 1", "Launchpad Mini"]
    }
  }
}
```

When a source is unavailable, includes `install_hint`:
```json
{
  "link": {
    "available": false,
    "name": "Ableton Link",
    "install_hint": "pip install aalink"
  }
}
```

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

**`change_scheduled`** (when ParameterScheduler queues a change):
```json
{
  "type": "change_scheduled",
  "delay_ms": 250
}
```

**`change_applied`** (when the scheduled change fires):
```json
{
  "type": "change_applied"
}
```

### Client → Server

Beat state forwarding (cloud mode) is sent as regular parameters via the data channel parameter update message. The keys `bpm`, `beat_phase`, `bar_position`, `beat_count`, `is_playing` are intercepted by `FrameProcessor.update_parameters()` and routed to `TempoSync.update_client_beat_state()`.

`quantize_mode` and `lookahead_ms` are also sent as regular parameters and intercepted by `FrameProcessor`.

---

## 5. Beat State Flow

### 5.1 Local Mode

```
Ableton Link / MIDI device
    │
    ▼
TempoSource._poll_loop() / ._listen_loop()
    │  Caches BeatState behind a lock
    ▼
TempoSync.get_beat_state()
    │  Returns source state (client state stale or absent)
    ├──────────────────────────────────────────────┐
    ▼                                              ▼
PipelineProcessor.__call__()                 _notification_loop()
    │  Injects bpm, beat_phase,                   │  15Hz push to all
    │  bar_position, beat_count,                   │  registered sessions
    │  is_playing into call_params                 ▼
    ▼                                     WebRTC data channel
Pipeline.__call__(**call_params)               │
    │  Pipeline uses beat state                    ▼
    │  for rendering decisions              Frontend useTempoSync
    ▼                                         updates UI state
Video output (WebRTC stream)
```

### 5.2 Cloud Mode

```
Browser (Link client / manual BPM)
    │
    ▼
WebRTC data channel parameter message
    │  Includes bpm, beat_phase, bar_position, beat_count, is_playing
    ▼
FrameProcessor.update_parameters()
    │  Calls tempo_sync.update_client_beat_state(params)
    │  Beat keys stripped from params, cached as BeatState(source="client")
    ▼
TempoSync.get_beat_state()
    │  Returns client state (fresh, < 2s old)
    ▼
PipelineProcessor.__call__()
    │  Same injection as local mode
    ▼
Pipeline.__call__(**call_params)
```

---

## 6. Parameter Scheduling

The ParameterScheduler solves a timing problem: if a VDJ changes the prompt and wants it to take effect on the next bar downbeat, the change needs to be applied *before* the downbeat to account for pipeline processing latency.

### Delay computation

Given the current beat state and quantize mode, the scheduler computes:

1. How many beats until the next boundary (depends on mode)
2. Convert to wall-clock seconds: `beats_until * (60 / bpm)`
3. Subtract lookahead: `delay = time_until_boundary - (lookahead_ms / 1000)`
4. If delay < 0 (lookahead exceeds boundary distance), wrap to the next cycle

### Near-boundary handling

When `beats_until < 0.01` (essentially on a boundary), the scheduler adds one full cycle to avoid scheduling for "right now" which would be indistinguishable from immediate.

### Merge semantics

When a timer is already pending:
- New params merge into the existing pending dict (last-write-wins for same keys)
- The original target boundary is preserved (no recomputation)
- Only one `change_applied` fires for the merged set

### Test coverage

`tests/test_parameter_scheduler.py` (594 lines) covers:
- Delay computation for all quantize modes (beat, bar, 2_bar, 4_bar)
- Odd time signatures (3/4, 7/8)
- Boundary edge cases (phase=0.0, phase=1.0, phase > 1.0, negative phase)
- Lookahead subtraction, lookahead exceeding boundary, very fast tempo
- Large beat counts (1,000,003)
- Timer firing and callback invocation
- Param merging and last-write-wins
- Cancel and re-schedule
- 50 concurrent schedule() calls from different threads
- Schedule during apply (slow callback)
- Invalid quantize mode rejection
- Negative lookahead clamping
- Notification callback exceptions

---

## 7. Demo Pipelines

### beat-viz

A pure-PyTorch beat visualizer with no model dependencies. Renders:
- Background with slowly rotating hue (keyed to beat_count)
- 24 radial bars with beat-driven wave animation
- Expanding center pulse ring on each beat
- 4 beat indicator dots at the bottom (downbeat highlighted in warm colors)
- BPM text in the top-right corner (3x5 bitmap font)

Useful for verifying that beat state is flowing correctly without loading any ML models.

### metronome

A visual metronome designed specifically for testing ParameterScheduler's lookahead compensation:
- Beat pulse flash (top third of frame)
- Beat indicator pips (bottom of frame)
- Beat/bar number display (center)
- BPM display (top-right)
- Three toggleable color layers (A=magenta, B=cyan, C=gold) that mix additively
- **Artificial latency** (`latency_ms` parameter) that simulates pipeline processing delay via `time.sleep()`

The workflow: set `latency_ms` to simulate a slow pipeline, toggle layers with quantize enabled, and adjust lookahead until the visual layer change lands exactly on the beat. This validates that the lookahead compensation works correctly.

---

## 8. Edge Cases and Robustness Analysis

### Strengths

- **Thread safety**: Separate locks for source access, client state, and notification sessions. BeatState is a frozen dataclass (immutable).
- **Graceful degradation**: Everything works if `aalink`/`mido` aren't installed. UI shows install hints. Pipelines that don't use beat kwargs are unaffected.
- **Cloud mode**: Client-forwarded beat state means the feature works in cloud deployments where the server has no Link/MIDI access.
- **Test coverage**: The ParameterScheduler has 594 lines of adversarial tests covering boundary math, concurrency, and edge cases.

### Issues to address before merging to main

**1. `threading.Timer` is not monotonic-clock-based.**
`threading.Timer` uses `time.sleep()` internally, which can be affected by system clock adjustments (NTP corrections, daylight savings). For sub-second musical timing, this is usually fine, but `time.monotonic()` would be more correct.

*Severity: Low. In practice, NTP corrections are small enough that the error is inaudible/invisible.*

**2. Link reconnection is silent.**
If all Link peers disconnect, `num_peers` drops to 0 but the source keeps running. If the Link session is destroyed and recreated (e.g., Ableton restarts), `aalink` should reconnect automatically, but there's no notification to the UI or logging of peer count changes.

*Recommendation: Log peer count changes. Consider a "Link peers: 0" warning in the UI.*

**3. MIDI clock jitter.**
EMA with alpha=0.15 provides moderate smoothing. With unstable MIDI interfaces (USB hubs, high-latency drivers), the BPM readout may jitter. Some MIDI setups send timing in bursts.

*Recommendation: Add a configurable smoothing factor or a separate "MIDI jitter tolerance" setting. Consider a median-of-N filter as an alternative to EMA for better outlier rejection.*

**4. Client beat state staleness threshold is hardcoded to 2 seconds.**
`CLIENT_BEAT_STATE_STALE_SECONDS = 2.0` is not configurable. In high-latency cloud environments, 2 seconds may be too short; in local setups with WebRTC data channel, it could be tighter.

*Recommendation: Make configurable via environment variable or API parameter.*

**5. `beat_viz` hardcodes `BEATS_PER_BAR = 4`.**
The pipeline ignores `TempoSync.beats_per_bar`. If a user sets 3/4 or 7/8 time, the beat dots at the bottom always show 4.

*Recommendation: Pass `beats_per_bar` as an additional kwarg or derive from `bar_position`.*

**6. No tap tempo.**
VDJs who don't have Link or MIDI sometimes need to tap a BPM manually. There's no tap tempo UI or API.

*Recommendation: Add a tap tempo mode as a third "source" -- purely frontend-driven, setting an internal BPM.*

**7. `bar_position` inconsistency.**
In the Link source, `bar_position = beat % beats_per_bar` which gives a float (e.g., 2.75 = beat 3 at 75% phase). In the MIDI source, `bar_position = (tick_count % bar_ticks) / PPQN` which is conceptually the same but computed differently. The `beat_viz` pipeline uses `int(bar_position) % BEATS_PER_BAR` which depends on this being consistent.

*Recommendation: Add a clear specification for `bar_position` in the BeatState docstring and add assertions in tests to verify both sources produce equivalent values.*

**8. No `is_playing` handling in ParameterScheduler.**
If `is_playing` is false (transport stopped), the scheduler still computes delays from beat state. It should probably pause or apply immediately.

*Recommendation: Check `is_playing` in `schedule()` and apply immediately when transport is stopped.*

**9. Notification loop error handling.**
If a `NotificationSender.call()` raises (data channel closed, etc.), the exception is silently swallowed. The session stays in `_notification_sessions` and continues receiving (failed) calls until unregistered.

*Recommendation: Auto-unregister sessions that fail N consecutive times.*

**10. ParameterScheduler has no way to flush on disable.**
If a user disables tempo sync while a timer is pending, the timer still fires and applies the params. `TempoSync.disable()` doesn't cancel the scheduler.

*Recommendation: Call `parameter_scheduler.cancel_pending()` when tempo sync is disabled.*

---

## 9. Modulation Branch Analysis

The `marco/feat/beatsync-modulation` branch (2 commits ahead of `beatsync`) attempts to add pipeline-level beat modulation effects. Here's a detailed assessment.

### 9.1 What it adds

**`BeatNoiseModulationBlock`** (241 lines) -- a diffusers `ModularPipelineBlocks` block that:
- Detects beat onsets via `beat_count` change detection
- Uses a cosine envelope with exponential decay for the breathing curve
- Modulates `noise_scale` (primary lever -- controls first denoising timestep) and `kv_cache_attention_bias` (secondary lever -- suppresses past-frame attention on beats)
- Applies downbeat accent (100% on beat 1, 85% on beat 3, 70% on beats 2/4)
- Has an `intensity` parameter (0.0-1.0) for user control

**`TempoEffectsBlock`** (483 lines) -- an expanded multi-effect system replacing `BeatNoiseModulationBlock`:
1. **Noise Breathing** -- oscillates noise_scale and kv_cache_attention_bias (same as above but configurable envelope shape and accent pattern)
2. **Prompt Cycling** -- switches prompts on beat boundaries with modes: sequential, random, pingpong
3. **Ref Image Switching** -- cycles reference images (VACE or first_frame) on beat boundaries
4. **Denoising Modulation** -- varies denoising step count with beat
5. **VACE Context Pulse** -- pulses vace_context_scale on beat

**`TempoEffectsPanel`** (543 lines) -- new frontend component with:
- Collapsible sections for each effect
- Per-effect enable/disable toggles
- Intensity sliders, envelope shape selectors, accent pattern selectors
- Prompt list management for prompt cycling
- Cycle mode selectors (sequential, random, pingpong)
- Beat interval controls

### 9.2 What it deletes

- `ParameterScheduler` (201 lines) -- the entire beat-quantized scheduling system
- `tests/test_parameter_scheduler.py` (593 lines) -- all scheduler tests
- `MetronomePipeline` and its schema -- the latency testing tool
- `LogPanel` component and `useLogStream` hook (unrelated to tempo)
- Quantize logic from `StreamPage` (the `isQuantizeActive` flag and discrete param tagging)
- `SettingsState.quantizeMode` and `SettingsState.lookaheadMs` type definitions

### 9.3 Problems

**1. Deleting the ParameterScheduler is wrong.**
The scheduler and modulation effects serve different purposes. The scheduler handles discrete parameter changes (prompt switch, layer toggle) landing on beat boundaries. The modulation effects handle continuous per-frame parameter oscillation. Both are needed. The modulation branch collapses them into one system (everything happens inside the pipeline block), losing the clean separation.

**2. Tight coupling to LongLive's ModularPipelineBlocks.**
`TempoEffectsBlock` inherits from `ModularPipelineBlocks` and uses `get_block_state()` / `set_block_state()`. This only works with the LongLive pipeline's diffusers-based modular block system. Other pipelines (StreamDiffusionV2, RewardForcing, MemFlow, etc.) cannot use these effects.

**3. The `tempo_effects` dict is sent as an unvalidated blob.**
The frontend sends the entire effect config as a nested dict through the data channel. There's no Pydantic schema validation, no type checking, and no error handling for malformed configs. A typo in an envelope name silently uses the default.

**4. Mixed concerns in one branch.**
The branch deletes LogPanel, auth code, cloud proxy code -- changes unrelated to tempo. This makes it impossible to cherry-pick the useful modulation work.

**5. No tests.**
All 593 lines of scheduler tests were deleted. No new tests were added for the modulation effects.

**6. Beat tracking at low FPS is fundamentally limited.**
The pipeline runs at ~2-7 FPS. At 120 BPM (500ms beats), the pipeline processes one chunk every ~150-500ms. Smooth envelopes are impossible at this rate -- you get at most 1-3 samples per beat. The `TempoEffectsBlock` acknowledges this ("smooth envelope is impossible to track at this rate") and falls back to binary beat detection with exponential decay. This is the correct approach, but the envelope shape selector (cosine/exponential/square) in the frontend is misleading -- at pipeline FPS, these shapes are indistinguishable.

### 9.4 What to salvage

Despite the structural issues, the modulation branch contains good ideas:

1. **Beat onset detection via `beat_count` change** -- reliable even at low FPS
2. **The effect taxonomy** -- noise breathing, prompt cycling, ref image switching, denoising modulation, VACE context pulse are exactly the right set of VDJ-relevant effects
3. **Downbeat/backbeat accent patterns** -- musically meaningful emphasis
4. **Prompt cycling modes** (sequential, random, pingpong) -- VDJs will want all three
5. **The `TempoEffectsPanel` UI structure** -- collapsible per-effect sections with enable toggles is the right UX pattern

---

## 10. Recommendations for Main

### 10.1 Merge the beatsync branch as-is

The `marco/feat/beatsync` branch is clean and ready for main with minor fixes:

- Fix `beat_viz` to respect `beats_per_bar` instead of hardcoding 4
- Add `cancel_pending()` call to `TempoSync.disable()` (coordinate with FrameProcessor)
- Add `is_playing` check to `ParameterScheduler.schedule()`
- Document `bar_position` semantics in `BeatState` docstring

These are small, low-risk changes that can be done in the same PR.

### 10.2 Redesign modulation as a pipeline-agnostic system

The modulation effects should be a separate feature built on top of the beatsync infrastructure, not a replacement for it. Key design principles:

1. **Keep ParameterScheduler** for discrete changes (prompt switches, layer toggles)
2. **Add a ModulationEngine** at the server level (in `FrameProcessor` or a new `modulation.py`) that operates on parameters *before* they're passed to the pipeline
3. **Make effects pipeline-agnostic** -- they modify kwargs, not diffusers block state
4. **Validate configs with Pydantic schemas** at the API level
5. **Add comprehensive tests** for every effect

### 10.3 Proposed modulation architecture

```
Frontend (TempoEffectsPanel)
    │
    │  POST /api/v1/tempo/effects  (Pydantic-validated config)
    ▼
FrameProcessor
    │
    ├── ParameterScheduler  (discrete changes: prompt switch, etc.)
    │
    └── ModulationEngine    (continuous effects: breathing, pulsing)
            │
            │  On each pipeline call:
            │  1. Get current BeatState
            │  2. For each enabled effect, compute modulated value
            │  3. Override call_params with modulated values
            ▼
        PipelineProcessor
            │  Injects beat state + modulated params
            ▼
        Pipeline.__call__(**call_params)
```

**ModulationEngine** responsibilities:
- Holds a list of active modulation effects
- Each effect is a simple function: `(beat_state, config, current_params) -> param_overrides`
- Effects are registered by name and applied in order
- Config is validated by Pydantic schemas when received from the API
- Effects operate on the same kwargs that pipelines receive -- no diffusers-specific coupling

**Effect implementations** (one function each, not ModularPipelineBlocks):
- `noise_breathing(beat_state, config, params)` -- modulates `noise_scale`
- `prompt_cycling(beat_state, config, params)` -- switches `prompts` based on beat interval
- `denoising_modulation(beat_state, config, params)` -- modulates `denoising_step_list[0]`
- `vace_context_pulse(beat_state, config, params)` -- pulses `vace_context_scale`
- `ref_image_cycling(beat_state, config, params)` -- cycles `vace_ref_images`

This design means any pipeline that accepts `noise_scale` or `prompts` as kwargs gets modulation for free. It also means the modulation system is testable in isolation (pure functions on dicts).

---

## 11. Feature Gaps and Future Work

### Near-term (should be in the initial main merge or fast-follow)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Tap tempo** | Frontend-only BPM source for users without Link/MIDI. Tap a button N times, compute average interval. | High |
| **OSC input** | Many VDJ setups use OSC for parameter control. Accept BPM/beat from OSC messages. | Medium |
| **`beats_per_bar` in pipelines** | Pass as an additional kwarg so pipelines can render correct time signatures. | High |
| **Modulation engine** | Pipeline-agnostic parameter modulation as described in 10.3. | High |
| **Tempo API docs** | OpenAPI/Swagger documentation for the REST endpoints. | Medium |

### Medium-term

| Feature | Description |
|---------|-------------|
| **Multi-rate modulation** | Different effects at different beat divisions (1/4 note breathing, 1-bar prompt switch, 4-bar ref image cycle) |
| **MIDI note/CC mapping** | Map MIDI notes or CC to parameter changes (not just clock) |
| **Beat division** | Support half-beat, triplet, and dotted-note divisions for finer modulation |
| **Modulation presets** | Save/load effect configurations as presets |
| **Audio analysis** | Extract beat from audio input (for when there's no Link/MIDI but there is audio) |
| **Automation lanes** | Timeline-based modulation curves synced to beat grid |
| **LFO-style modulation** | Sine/triangle/saw/square LFOs synced to beat divisions for parameter modulation |
| **Transport control** | Play/stop/reset beat counter from Scope UI |

### Long-term

| Feature | Description |
|---------|-------------|
| **Resolume Wire protocol** | Direct integration with Resolume's parameter system |
| **Ableton Max for Live device** | Control Scope parameters from Ableton |
| **Multi-session sync** | Multiple Scope instances sharing beat state and prompt sequences |

---

## 12. File Index

### Backend

| File | Lines | Role |
|------|-------|------|
| `src/scope/server/tempo_sync.py` | 348 | Central tempo manager, BeatState, TempoSource ABC |
| `src/scope/server/parameter_scheduler.py` | 202 | Beat-quantized parameter scheduling |
| `src/scope/server/tempo_sources/__init__.py` | 2 | Package init |
| `src/scope/server/tempo_sources/link.py` | 113 | Ableton Link adapter (aalink) |
| `src/scope/server/tempo_sources/midi_clock.py` | 152 | MIDI clock adapter (mido/rtmidi) |
| `src/scope/server/app.py` | 80* | REST endpoints (lines 2230-2310) |
| `src/scope/server/frame_processor.py` | ~30* | TempoSync/scheduler integration |
| `src/scope/server/pipeline_processor.py` | ~10* | Beat state injection into call_params |
| `src/scope/server/webrtc.py` | ~20* | Session registration for tempo notifications |
| `src/scope/server/schema.py` | 52* | Pydantic models for tempo API |

*\* Lines directly related to tempo within larger files.*

### Frontend

| File | Lines | Role |
|------|-------|------|
| `frontend/src/hooks/useTempoSync.ts` | 175 | Tempo state management hook |
| `frontend/src/components/settings/TempoSyncSection.tsx` | 338 | Tempo sync UI controls |
| `frontend/src/lib/api.ts` | ~90* | API types and functions |
| `frontend/src/hooks/useUnifiedWebRTC.ts` | ~30* | Data channel tempo message handling |
| `frontend/src/pages/StreamPage.tsx` | ~60* | Quantize integration, tempo wiring |
| `frontend/src/types/index.ts` | ~5* | quantizeMode, lookaheadMs types |

### Pipelines

| File | Lines | Role |
|------|-------|------|
| `src/scope/core/pipelines/beat_viz/__init__.py` | 6 | Package init |
| `src/scope/core/pipelines/beat_viz/schema.py` | 24 | Config schema |
| `src/scope/core/pipelines/beat_viz/pipeline.py` | 239 | Beat visualizer (PyTorch) |
| `src/scope/core/pipelines/metronome/__init__.py` | -- | Package init |
| `src/scope/core/pipelines/metronome/schema.py` | 63 | Config with latency_ms, layers |
| `src/scope/core/pipelines/metronome/pipeline.py` | 221 | Visual metronome for latency testing |

### Tests

| File | Lines | Role |
|------|-------|------|
| `tests/test_parameter_scheduler.py` | 594 | Adversarial scheduler tests |

### Examples

| File | Role |
|------|------|
| `examples/tempo-drum-machine/index.html` | Standalone HTML tempo API demo with Web Audio |

### Dependencies

```toml
# pyproject.toml optional dependencies
[project.optional-dependencies]
link = ["aalink>=0.1.1"]
midi = ["mido>=1.3.0", "python-rtmidi>=1.5.0"]
```
