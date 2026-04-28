# Using OSC

Scope exposes its parameters over [OSC (Open Sound Control)](https://opensoundcontrol.stanford.edu/) so external tools — TouchDesigner, Resolume, MaxMSP, hardware controllers, custom Python scripts — can drive the running graph in real time. The OSC server is always on; it shares a UDP socket with the HTTP API on the same port (default `8000`).

## Table of Contents

- [Quick Start](#quick-start)
- [What's Reachable via OSC](#whats-reachable-via-osc)
- [Configure OSC per Node](#configure-osc-per-node)
- [Address Format](#address-format)
- [Discovering Paths](#discovering-paths)
- [Defaults in the Description](#defaults-in-the-description)
- [Validation](#validation)
- [Routing Internals](#routing-internals)
- [TouchDesigner Setup](#touchdesigner-setup)
- [Python Examples](#python-examples)
- [REST API](#rest-api)
- [Limitations](#limitations)

---

## Quick Start

1. Start Scope (`uv run daydream-scope`). The OSC server starts automatically and listens on UDP `8000`.
2. Send a message:

   ```python
   from pythonosc.udp_client import SimpleUDPClient
   client = SimpleUDPClient("127.0.0.1", 8000)
   client.send_message("/scope/prompt", "a beautiful sunset over the ocean")
   client.send_message("/scope/noise_scale", 0.5)
   ```

3. Open the live OSC reference at `http://localhost:8000/api/v1/osc/docs` to see every address, type, range, default, and a copy-paste example.

> [!NOTE]
> Set the OSC port via the `SCOPE_PORT` environment variable. UDP and HTTP coexist on the same port, so changing one changes both.

---

## What's Reachable via OSC

Scope exposes three layers of OSC paths:

| Layer | Source | Default address | Notes |
|---|---|---|---|
| **Runtime globals** | Built-in (`prompt`, `noise_scale`, `paused`, `manage_cache`, `reset_cache`, `transition_steps`, `interpolation_method`, …) | `/scope/<param>` | Always on. Backwards-compatible with pre-existing TouchDesigner setups. |
| **Pipeline runtime params** | Each loaded pipeline's `is_load_param=False` fields | `/scope/<param>` (default) or `/scope/<node>/<param>` (per-instance, opt-in) | One bare-address entry per param while no per-node override is set; switches to namespaced when you customize the node. |
| **Graph nodes** (Source / Sink / Slider / XYPad / Bool / Trigger / Tempo / Output / Note / Primitive) | The graph editor's per-node `oscConfig` | `/scope/<node>/<param>` | Off by default; opted in via right-click → **Configure OSC…** |

The Source / Sink / UI-node layer is the new piece — previously only pipeline params could be reached.

---

## Configure OSC per Node

The graph editor's right-click menu has a **"Configure OSC…"** item:

1. Right-click any node → **Configure OSC…**
2. The modal lists every OSC-eligible param for that node type. Each row has:

   | Column | Behavior |
   |---|---|
   | **Expose** | Tick to publish this param's address. Off by default (except for pipeline runtime params, which are auto-exposed at the legacy flat address until you customize anything). |
   | **Address** | Auto-fills as `/scope/<node-slug>/<param>`. Editable — paste any address you want (e.g. `/scope/tempo`, `/scope/main/prompt`). |
   | **Default** | Advisory metadata published in the OSC docs so external clients can mirror Scope's starting state. Defaults to the node's current value. |

3. **Save**. The graph re-publishes its OSC inventory to the backend within ~300ms; the new address becomes reachable immediately.

`oscConfig` is stored on the node and round-trips with the rest of the graph (saving, exporting, re-importing all preserve it).

> [!NOTE]
> A param set's default is **advisory** — it appears in the description so OSC clients can initialize their UI to match Scope, but Scope does not auto-emit it on session start. Auto-apply is a deliberate follow-up.

### What to expose, by node type

| Node | Exposable params |
|---|---|
| Source | `sourceMode` (enum), `sourceFlipVertical` (bool) |
| Output | `outputSinkEnabled` (bool), `outputSinkType` (enum) |
| Slider | `value` (float) |
| XY Pad | `padX`, `padY` (float) |
| Bool | `value` (bool) |
| Trigger | `value` (bool — send `true` to fire) |
| Tempo | `tempoBpm` (float, 20–999), `tempoEnabled` (bool) |
| Primitive | `value` (string) |
| Note | `noteText` (string) |
| Pipeline | every `is_load_param=False` field from the pipeline's schema |

Composite-shape params (knobs[], MIDI channels[], tuple values[]) are intentionally not exposed in the MVP — they need a richer addressing scheme.

---

## Address Format

```
/scope/<node-slug>/<param>
```

`<node-slug>` is derived from the node's display title (the user-editable header), slugified to kebab-case. Falls back to the React Flow node id when the title has no slug-able characters.

Examples:

| Node title | Field | Default address |
|---|---|---|
| `Tempo` (Slider) | `value` | `/scope/tempo/value` |
| `Source` | `sourceMode` | `/scope/source/sourceMode` |
| `Main` (Pipeline) | `prompt` | `/scope/main/prompt` |
| `Secondary` (Pipeline) | `prompt` | `/scope/secondary/prompt` |

> [!IMPORTANT]
> If two nodes resolve to the same slug (two sliders both titled "Tempo"), the most recently re-saved one wins. Rename one or override the address explicitly to avoid collisions. A UI warning is planned.

### Backwards compatibility

Pipeline runtime params keep their **flat** address `/scope/<param>` until you open Configure OSC on the pipeline node and save any change. Existing rigs sending `/scope/prompt`, `/scope/noise_scale`, `/scope/paused`, etc. continue to work without graph-side configuration.

The moment a user opts a pipeline param in (or out) explicitly, the legacy flat alias is replaced by the user's namespaced address for that node. This is what enables driving two pipeline instances independently when they share param names.

---

## Discovering Paths

### In-app

**Settings → OSC** has a "Currently exposed paths" panel. Each row is click-to-copy.

### HTML reference

Open [`http://localhost:8000/api/v1/osc/docs`](http://localhost:8000/api/v1/osc/docs) for an auto-generated reference page. Every path includes its address, type, constraints (min / max / enum), default, and a one-click Python snippet you can paste into TouchDesigner's text DAT.

### Programmatic

```bash
curl -s http://localhost:8000/api/v1/osc/paths | jq
```

Response shape:

```json
{
  "active": {
    "Runtime": [
      { "key": "prompt", "type": "string", "osc_address": "/scope/prompt", … }
    ],
    "streamdiffusionv2": [
      { "key": "noise_scale", "type": "float", "min": 0.0, "max": 1.0, "default": 0.7, "osc_address": "/scope/noise_scale", … }
    ],
    "Tempo": [
      { "osc_address": "/scope/tempo/value", "type": "float", "default": 1.0, "node_id": "slider-1", "param": "value", … }
    ]
  },
  "available": { … },
  "active_pipeline_ids": ["streamdiffusionv2"]
}
```

`active` groups everything currently reachable. `available` lists pipelines that exist in the registry but aren't loaded yet — their addresses become reachable as soon as the pipeline is loaded.

---

## Defaults in the Description

Every path entry that has a default value carries it in two places:

- **`/api/v1/osc/paths`** — `default` field on the entry.
- **`/api/v1/osc/docs`** — Default column in the rendered table.

Defaults come from (in order):

1. The user-set value in **Configure OSC…** (per node).
2. The node's current `data.<param>` value (e.g. the slider's current position).
3. The pipeline schema's `default` (for pipeline runtime params).

This lets external clients initialize their UI to match Scope's starting state without an extra round-trip.

---

## Validation

The OSC server validates every incoming message against the path's type / min / max / enum constraints before broadcasting. Invalid messages are logged but never reach the pipeline. Toggle **Settings → OSC → Log Messages** to see all messages (valid + invalid) in the Scope logs.

| Type | Accepts | Rejection example |
|---|---|---|
| `float` / `number` | `int` or `float` | string → "type mismatch" |
| `integer` | `int` | float → "type mismatch" |
| `bool` / `boolean` | `bool` / `int` / `float` (truthy = on) | string → "type mismatch" |
| `string` | `str` | int → "type mismatch" |
| `integer_list` | non-empty list of ints | list with non-ints → "type mismatch: item N of type …" |

Out-of-range numeric values are rejected with `"value X below minimum Y"` / `"above maximum Y"`. Enum-violating strings get `"value 'foo' not in allowed values […]"`.

---

## Routing Internals

The OSC server splits incoming messages into three routing buckets based on the matched path entry:

1. **Pipeline node** (entry has both `node_id` and `pipeline_id`) — broadcasts `{node_id, <param>: value}` to all WebRTC sessions; `frame_processor.update_parameters` routes it to the matching pipeline processor.
2. **UI-only node** (entry has `node_id` but no `pipeline_id`) — emits an SSE `osc_command` event the frontend listens to; the React Flow state for the matching node updates locally. No backend processor is involved (Slider / Bool / etc. are frontend-only state).
3. **Registry-derived flat path** (entry has no `node_id`) — broadcasts `{<key>: value}` to all WebRTC sessions, matching legacy behavior.

In all three cases the SSE stream is also fanned out so the frontend can mirror the param change in the UI.

---

## TouchDesigner Setup

1. Add an **OSC Out CHOP** (or **OSC Out DAT** for strings).
2. Set **Network Address** to `127.0.0.1` (or the IP of the machine running Scope).
3. Set **Network Port** to `8000`.
4. Add a channel with the address you want to drive — `/scope/prompt`, `/scope/noise_scale`, `/scope/tempo/value`, etc.
5. Animate or bind the channel value to your TD parameters.

> [!TIP]
> Click the address row in `http://localhost:8000/api/v1/osc/docs` to copy a working Python snippet. Paste it into a TD **Text DAT** for offline testing before wiring up CHOPs.

---

## Python Examples

```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8000)

# Drive the active pipeline's prompt
client.send_message("/scope/prompt", "a glowing reef at night")

# Animate noise scale
import time
for v in (0.0, 0.25, 0.5, 0.75, 1.0):
    client.send_message("/scope/noise_scale", v)
    time.sleep(0.5)

# Toggle a per-node Bool you've exposed at /scope/strobe/value
client.send_message("/scope/strobe/value", True)

# Trigger a node — the same as clicking it once in the UI
client.send_message("/scope/cue/value", True)
```

To listen for parameter changes that originate elsewhere (UI clicks, MIDI, other OSC senders), connect to the SSE stream:

```python
import json, requests

with requests.get("http://localhost:8000/api/v1/osc/stream", stream=True) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data:"):
            continue
        event = json.loads(line[len(b"data: "):])
        # event = {"type": "osc_command", "key": "tempo/value", "value": 0.85,
        #          "node_id": "slider-1", "param": "value"}
        print(event)
```

---

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/osc/status` | Listening state, port, host, log-verbosity flag |
| `PUT` | `/api/v1/osc/settings` | Toggle `log_all_messages` |
| `GET` | `/api/v1/osc/paths` | Active + available paths, JSON |
| `GET` | `/api/v1/osc/docs` | Self-contained HTML reference page |
| `GET` | `/api/v1/osc/stream` | Server-Sent Events stream of every received OSC command |
| `POST` | `/api/v1/osc/inventory` | (Internal) Replace the graph-supplied path inventory; called by the frontend whenever `oscConfig` changes |

---

## Limitations

- **Slug collisions** are not yet warned in the UI. Two nodes that share a derived slug share the address; the most recently registered wins.
- **Composite params** (knobs[], midiChannels[], tupleValues[]) are skipped for now.
- **Auto-apply of defaults at session start** is intentionally not implemented; the per-param default is description-only metadata.
- **HDR pipeline params** that don't fit the float/int/bool/string/integer-list type system aren't reachable via OSC.
- **OSC port** is shared with the HTTP server. To run multiple Scope instances on one machine, give each a different `SCOPE_PORT`.
