# Tempo Sync

Tempo sync allows Scope's real-time video pipelines to lock to an external beat clock so visual output reacts to music in time. It supports [Ableton Link](https://www.ableton.com/en/link/) and MIDI clock as tempo sources.

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Tempo Sources](#tempo-sources)
- [Beat-Quantized Parameters](#beat-quantized-parameters)
- [Modulation](#modulation)
- [Beat Cache Reset](#beat-cache-reset)
- [Prompt Cycling](#prompt-cycling)
- [Demo Pipelines](#demo-pipelines)
- [Cloud Mode](#cloud-mode)
- [REST API](#rest-api)
- [Architecture Overview](#architecture-overview)

---

## Installation

Tempo sync requires optional dependencies depending on which tempo source you want to use.

### Ableton Link

```bash
uv sync --extra link
```

This installs [aalink](https://pypi.org/project/aalink/), an async Python wrapper for Ableton Link.

### MIDI Clock

```bash
uv sync --extra midi
```

This installs [mido](https://pypi.org/project/mido/) and [python-rtmidi](https://pypi.org/project/python-rtmidi/).

**Linux only:** The ALSA library is required for MIDI. Install it with your package manager:

| Distro | Package |
|--------|---------|
| Debian / Ubuntu | `sudo apt install libasound2` |
| Fedora / RHEL | `sudo dnf install alsa-lib` |
| Arch | `sudo pacman -S alsa-lib` |

Docker images do not include ALSA since MIDI requires local hardware access.

### Both

```bash
uv sync --extra link --extra midi
```

If neither dependency group is installed, the tempo sync UI will show install hints and the feature will be unavailable.

---

## Getting Started

1. **Install dependencies** for your preferred tempo source (see above).
2. **Start the server:**
   ```bash
   uv run daydream-scope
   ```
3. **Open the Tempo Sync section** in the settings panel.
4. **Select a source** (Link or MIDI) and click the toggle to enable.
5. If using **Ableton Link**, any Link-enabled app on the same network (Ableton Live, Resolume, etc.) will automatically sync. You can also set the BPM directly from Scope to become the session tempo leader.
6. If using **MIDI Clock**, select your MIDI device from the dropdown (or let it auto-select the first available device).

Once enabled, you'll see a live beat indicator and the current BPM. The beat state (`bpm`, `beat_phase`, `bar_position`, `beat_count`, `is_playing`) is injected into every pipeline call automatically.

---

## Tempo Sources

### Ableton Link

- Joins the shared Link session on the local network automatically.
- Any Link-enabled app (Ableton Live, Resolume, Traktor, etc.) will see Scope as a peer.
- You can set the BPM from Scope to become the session tempo leader.
- The UI shows the number of connected Link peers.

### MIDI Clock

- Receives MIDI clock messages from an external device or DAW.
- Select a specific MIDI device from the dropdown, or let Scope auto-select the first available one.
- BPM is derived from incoming clock messages using smoothing for stability.
- Handles start, stop, and continue transport messages.

---

## Beat-Quantized Parameters

When tempo sync is active, parameter changes (like switching prompts or adjusting denoising strength) can be quantized to land on beat boundaries instead of applying immediately.

### Quantize Mode

Set the quantize mode in the Tempo Sync section:

| Mode | Behavior |
|------|----------|
| Immediate | Changes apply instantly (no quantization) |
| Beat | Changes snap to the next beat |
| Bar | Changes snap to the next bar boundary |
| 2 Bars | Changes snap to the next 2-bar boundary |
| 4 Bars | Changes snap to the next 4-bar boundary |

### Lookahead

Video pipelines have processing latency — a parameter change applied at the exact beat boundary may appear late on screen. The **lookahead** slider (0–1000ms) compensates by applying changes slightly before the boundary so the visual result lands on the beat.

**How to calibrate:**

1. Load the **metronome** pipeline.
2. Set a simulated latency with the `latency_ms` parameter.
3. Enable quantization and toggle a color layer.
4. Adjust the lookahead slider until the visual change lands exactly on the beat.

---

## Modulation

Modulation continuously varies numeric pipeline parameters on every frame, driven by the beat phase. This creates rhythmic visual effects — for example, pulsing the noise scale on every beat or sweeping denoising steps across a bar.

### Modulatable Parameters

Not all parameters support modulation. Pipelines expose modulatable parameters in their schema. Currently supported:

| Pipeline | Parameters |
|----------|-----------|
| LongLive | `noise_scale`, `denoising_steps` |
| Krea | `denoising_steps` |
| MemFlow | `denoising_steps` |
| RewardForcing | `denoising_steps` |
| StreamDiffusionV2 | `denoising_steps` |

The Modulation section in the UI automatically shows only the parameters available for the currently loaded pipeline.

### Modulation Controls

For each modulatable parameter, you can configure:

- **Shape** — The waveform used to drive the modulation:
  - `sine`, `cosine`, `triangle`, `saw`, `square` — oscillate around a base value
  - `exp_decay` — pulse at the start of each cycle, then decay
- **Depth** (0–100%) — How much the parameter varies from its base value.
- **Rate** — How fast the modulation cycles:
  - `half_beat` (2x per beat), `beat`, `2_beat`, `bar`, `2_bar`, `4_bar`
- **Base value** — The center value to modulate around (defaults to the parameter's current value).
- **Min / Max** — Optional bounds to clamp the modulated value.

---

## Beat Cache Reset

Some pipelines use cached state between frames. Beat cache reset regenerates the cache seed on beat boundaries, creating a rhythmic "reset" effect. Set the rate in the Tempo Sync section:

| Rate | Effect |
|------|--------|
| Off | No beat-synced resets |
| Beat | Reset every beat |
| Bar | Reset every bar |
| 2 Bars | Reset every 2 bars |
| 4 Bars | Reset every 4 bars |

---

## Prompt Cycling

When you have multiple prompts in the timeline, prompt cycling automatically advances through them on beat boundaries. Set the cycle rate in the Tempo Sync section to control how often prompts advance.

---

## Demo Pipelines

Two pipelines are included specifically for testing and visualizing tempo sync:

### Metronome

A visual metronome for calibrating the lookahead setting:

- Displays beat pulses, beat/bar numbers, and current BPM.
- Has three color layers (A/B/C) that mix additively — toggle them with quantization enabled to test timing.
- Includes a `latency_ms` parameter to simulate pipeline processing delay.

### ModScope

An oscilloscope that visualizes modulation output in real time. Use it to verify that your modulation settings produce the expected waveforms.

---

## Cloud Mode

When running Scope in the cloud (without direct access to Link or MIDI devices), the browser can forward beat state to the server over the WebRTC data channel. This allows cloud deployments to stay beat-synced with a local Link session running in the browser.

---

## REST API

Tempo sync exposes REST endpoints under `/api/v1/tempo/`:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/tempo/status` | Current tempo status and beat state |
| `POST` | `/api/v1/tempo/enable` | Enable tempo sync |
| `POST` | `/api/v1/tempo/disable` | Disable tempo sync |
| `POST` | `/api/v1/tempo/set_tempo` | Set BPM (Link only) |
| `GET` | `/api/v1/tempo/sources` | Available tempo sources |

### Enable Request

```json
{
  "source": "link",
  "bpm": 120.0,
  "beats_per_bar": 4
}
```

For MIDI, use `"source": "midi_clock"` and optionally specify `"midi_device": "DeviceName"`.

---

## Architecture Overview

The tempo sync system is organized into five layers:

```
┌─────────────────────────────────────────────────────────┐
│                    TEMPO SOURCES                        │
│   Ableton Link  |  MIDI Clock  |  Client-Forwarded     │
└────────┬────────────────┬──────────────┬────────────────┘
         │                │              │
         ▼                ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                  TEMPOSYNC MANAGER                       │
│   Unified BeatState · Thread-safe · 15Hz notifications  │
└────────┬────────────────────────────────┬───────────────┘
         │                                │
         ▼                                ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ PARAMETER SCHEDULER  │   │     MODULATION ENGINE        │
│ Quantize changes to  │   │  Per-frame waveform-driven   │
│ beat/bar boundaries   │   │  parameter modulation        │
└──────────┬───────────┘   └──────────────┬───────────────┘
           └──────────┬───────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│               PIPELINE INJECTION                        │
│  Injects beat state + modulated params into pipeline    │
└─────────────────────────────────────────────────────────┘
```

**Key backend files:**

| File | Purpose |
|------|---------|
| `src/scope/server/tempo_sync.py` | TempoSync manager and BeatState |
| `src/scope/server/parameter_scheduler.py` | Beat-quantized parameter scheduling |
| `src/scope/server/modulation.py` | Modulation engine and config |
| `src/scope/server/tempo_sources/link.py` | Ableton Link adapter |
| `src/scope/server/tempo_sources/midi_clock.py` | MIDI clock adapter |
| `src/scope/server/pipeline_processor.py` | Beat injection and modulation apply |

**Key frontend files:**

| File | Purpose |
|------|---------|
| `frontend/src/hooks/useTempoSync.ts` | Tempo state management |
| `frontend/src/components/settings/TempoSyncSection.tsx` | Tempo sync UI controls |
| `frontend/src/components/settings/ModulationSection.tsx` | Modulation UI controls |
