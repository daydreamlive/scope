# Scope-Moondream Interface

This document describes the interface between the Scope pipeline system and the Moondream vision language model, including data flow diagrams for both local and cloud relay modes.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Scope Frontend (React)                       │
│                                                                     │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────────┐ │
│  │ Pipeline      │  │ Settings Panel    │  │ Input & Controls     │ │
│  │ Dropdown      │  │ (configuration)   │  │ (input category)     │ │
│  │               │  │                   │  │                      │ │
│  │ [Moondream ▼] │  │ Feature     [▼]   │  │ Question    [______] │ │
│  │               │  │ Caption Len [▼]   │  │ Detect Obj  [______] │ │
│  └──────────────┘  │ Temperature ─●──  │  └──────────────────────┘ │
│                     │ Max Objects ─●──  │                           │
│                     │ Infer Intv  ─●──  │  ┌──────────────────────┐ │
│                     │ Overlay Op  ─●──  │  │ WebRTC Video Stream  │ │
│                     │ Font Scale  ─●──  │  │ ┌──────────────────┐ │ │
│                     │ Compile     [○]   │  │ │ Annotated frames │ │ │
│                     └───────────────────┘  │ │ with overlays    │ │ │
│                                            │ └──────────────────┘ │ │
│                                            └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
         │                                           ▲
         │ Settings & params (HTTP or WebSocket)     │ Video frames (WebRTC)
         ▼                                           │
┌─────────────────────────────────────────────────────────────────────┐
│                       Scope Backend (FastAPI)                       │
│                                                                     │
│  ┌──────────────────┐  ┌───────────────┐  ┌──────────────────────┐ │
│  │ Pipeline Manager  │  │ Pipeline      │  │ Frame Processor      │ │
│  │                   │  │ Processor     │  │                      │ │
│  │ load("moondream") │──│ __call__()    │──│ WebRTC stream out    │ │
│  │ + merged_params   │  │ per frame     │  │ [0,255] uint8        │ │
│  └──────────────────┘  └───────────────┘  └──────────────────────┘ │
│            │                    │                                    │
│            │ __init__(**kwargs) │ __call__(**kwargs)                 │
│            ▼                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   MoondreamPipeline                          │   │
│  │  ┌─────────────────┐     ┌──────────────────────────────┐   │   │
│  │  │ Moondream2 Model │     │ Overlay Drawing (PIL)        │   │   │
│  │  │ (HuggingFace)    │────▶│ Boxes / Points / Text        │   │   │
│  │  └─────────────────┘     └──────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Frame Processing

```
                      INPUT                          OUTPUT
                 ┌──────────────┐              ┌──────────────┐
  Video Source   │ torch.Tensor │   Pipeline   │ torch.Tensor │   WebRTC
  (camera/file)──│ (1,H,W,C)   │──▶ __call__ ─│ (1,H,W,C)   │──▶ Stream
                 │ [0, 255]     │              │ [0, 1]       │
                 │ uint8        │              │ float32      │
                 └──────────────┘              └──────────────┘
                        │                             ▲
                        ▼                             │
                 ┌──────────────┐              ┌──────────────┐
                 │  tensor_to   │              │  pil_to      │
                 │  _pil()      │              │  _tensor()   │
                 └──────┬───────┘              └──────┬───────┘
                        │                             │
                        ▼                             │
                 ┌──────────────┐              ┌──────────────┐
                 │  PIL Image   │──▶ Moondream │  PIL Image   │
                 │  (H, W, RGB) │   Inference  │  + Overlays  │
                 └──────────────┘              └──────────────┘
```

---

## Pipeline Lifecycle

### Load Phase (once)

```
Frontend                Pipeline Manager              MoondreamPipeline
   │                          │                              │
   │ POST /pipeline/load      │                              │
   │ {pipeline_id:"moondream",│                              │
   │  load_params:{           │                              │
   │    compile_model: true   │                              │
   │  }}                      │                              │
   │─────────────────────────▶│                              │
   │                          │                              │
   │                          │  1. Get schema defaults      │
   │                          │     from MoondreamConfig     │
   │                          │     model_fields             │
   │                          │                              │
   │                          │  2. Merge:                   │
   │                          │     merged = {               │
   │                          │       feature: "detect",     │
   │                          │       temperature: 0.5,      │
   │                          │       compile_model: true,   │
   │                          │       ... all defaults ...   │
   │                          │     }                        │
   │                          │                              │
   │                          │  3. MoondreamPipeline(       │
   │                          │       **merged_params)       │
   │                          │─────────────────────────────▶│
   │                          │                              │
   │                          │                              │  Load vikhyatk/moondream2
   │                          │                              │  with torch.bfloat16
   │                          │                              │  to CUDA/CPU
   │                          │                              │
   │                          │                              │  model.compile()
   │                          │                              │  (if compile_model=True)
   │                          │                              │
   │                          │◀─ Pipeline instance ─────────│
   │◀── {message: "loaded"} ──│                              │
```

### Stream Phase (per frame)

```
Frame Processor         Pipeline Processor         MoondreamPipeline
     │                         │                          │
     │  frame from input_queue │                          │
     │  + runtime params       │                          │
     │────────────────────────▶│                          │
     │                         │                          │
     │                         │  pipeline(**{             │
     │                         │    video: [tensor],      │
     │                         │    feature: "detect",    │
     │                         │    detect_object: "cat", │
     │                         │    temperature: 0.5,     │
     │                         │    max_objects: 10,      │
     │                         │    inference_interval: 3,│
     │                         │    overlay_opacity: 0.8, │
     │                         │    font_scale: 1.0,      │
     │                         │  })                      │
     │                         │─────────────────────────▶│
     │                         │                          │
     │                         │                          │  ┌─ Frame N ───────────┐
     │                         │                          │  │                     │
     │                         │                          │  │ N % interval == 0?  │
     │                         │                          │  │   YES: run inference│
     │                         │                          │  │   NO:  use cache    │
     │                         │                          │  │                     │
     │                         │                          │  │ Draw overlays on    │
     │                         │                          │  │ PIL image           │
     │                         │                          │  │                     │
     │                         │                          │  │ Convert to tensor   │
     │                         │                          │  │ (1,H,W,C) [0,1]    │
     │                         │                          │  └─────────────────────┘
     │                         │                          │
     │                         │◀── {"video": tensor} ────│
     │                         │                          │
     │                         │  * 255 → [0,255] uint8   │
     │                         │  put in output_queue     │
     │◀────────────────────────│                          │
     │                                                    │
     │  encode + send via WebRTC                          │
```

---

## Moondream Model Interface

### Inputs and Outputs per Feature

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        model.caption(image, ...)                        │
│                                                                         │
│  Input:  PIL Image, length="short"|"normal"|"long"                     │
│          settings={temperature: float}                                  │
│                                                                         │
│  Output: {"caption": str}                                              │
│                                                                         │
│  Overlay: Semi-transparent text bar at bottom of frame                 │
│  ┌─────────────────────────────────────┐                               │
│  │                                     │                               │
│  │          (video frame)              │                               │
│  │                                     │                               │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                               │
│  │▓ A person sitting at a desk...    ▓│                               │
│  └─────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     model.query(image, question, ...)                   │
│                                                                         │
│  Input:  PIL Image, question=str                                       │
│          settings={temperature: float}                                  │
│                                                                         │
│  Output: {"answer": str}                                               │
│                                                                         │
│  Overlay: Q&A text bar at bottom of frame                              │
│  ┌─────────────────────────────────────┐                               │
│  │                                     │                               │
│  │          (video frame)              │                               │
│  │                                     │                               │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                               │
│  │▓ Q: How many people are there?    ▓│                               │
│  │▓ A: There are three people.       ▓│                               │
│  └─────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     model.detect(image, object, ...)                    │
│                                                                         │
│  Input:  PIL Image, object=str (e.g. "person")                         │
│          settings={max_objects: int}                                    │
│                                                                         │
│  Output: {"objects": [                                                 │
│             {"x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.9} │
│           ]}                                                           │
│          Coordinates normalized to [0, 1]                              │
│                                                                         │
│  Overlay: Green bounding boxes with numbered labels                    │
│  ┌─────────────────────────────────────┐                               │
│  │  ┌──#1──────┐                       │                               │
│  │  │          │    ┌──#2──┐           │                               │
│  │  │          │    │      │           │                               │
│  │  │          │    │      │           │                               │
│  │  │          │    └──────┘           │                               │
│  │  └──────────┘                       │                               │
│  └─────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     model.point(image, object, ...)                     │
│                                                                         │
│  Input:  PIL Image, object=str (e.g. "person")                         │
│          settings={max_objects: int}                                    │
│                                                                         │
│  Output: {"points": [                                                  │
│             {"x": 0.3, "y": 0.5}                                      │
│           ]}                                                           │
│          Coordinates normalized to [0, 1]                              │
│                                                                         │
│  Overlay: Red dots with numbered labels                                │
│  ┌─────────────────────────────────────┐                               │
│  │                                     │                               │
│  │      ●#1                            │                               │
│  │                   ●#2               │                               │
│  │                                     │                               │
│  │            ●#3                      │                               │
│  │                                     │                               │
│  └─────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Settings Propagation: Schema to UI

The Pydantic schema generates JSON that the frontend auto-renders into widgets. No frontend code changes are needed.

```
Backend (Python)                              Frontend (React)
================                              ================

MoondreamConfig                               schemaSettings.ts
  │                                             │
  │  Field(default=0.5,                         │  inferPrimitiveFieldType(prop)
  │        ge=0.0, le=1.0,                      │    │
  │        json_schema_extra=                    │    ├─ type=number + min+max → "slider"
  │          ui_field_config(order=3))           │    ├─ type=boolean          → "toggle"
  │                                              │    ├─ $ref to enum          → "enum"
  │                                              │    └─ type=string           → "text"
  ▼                                              ▼
Pydantic model_json_schema()                  Settings Panel / Input Panel
  │                                             │
  │  {                                          │  ┌──────────────────────┐
  │    "properties": {                          │  │ Feature      [▼]    │ ← enum
  │      "feature": {                           │  │ Caption Len  [▼]    │ ← enum
  │        "$ref": "#/$defs/MoondreamFeature",  │  │ Temperature  ─●──── │ ← slider
  │        "ui": {"order":1, "category":        │  │ Max Objects  ─●──── │ ← slider
  │               "configuration"}              │  │ Infer Intv   ─●──── │ ← slider
  │      },                                     │  │ Overlay Op   ─●──── │ ← slider
  │      "temperature": {                       │  │ Font Scale   ─●──── │ ← slider
  │        "type": "number",                    │  │ Compile      [○]    │ ← toggle
  │        "minimum": 0.0,                      │  ├──────────────────────┤
  │        "maximum": 1.0,                      │  │ Question     [____] │ ← text
  │        "default": 0.5,                      │  │ Detect Obj   [____] │ ← text
  │        "ui": {"order":3, "category":        │  └──────────────────────┘
  │               "configuration"}              │
  │      },                                     │  Fields without "ui" key
  │      "question": {                          │  are NOT rendered (hidden
  │        "type": "string",                    │  base class fields like
  │        "default": "What is in this image?", │  height, width, base_seed)
  │        "ui": {"order":1, "category":        │
  │               "input"}                      │
  │      }                                      │
  │    },                                       │
  │    "$defs": {                               │
  │      "MoondreamFeature": {                  │
  │        "enum": ["caption","query",          │
  │                 "detect","point"]            │
  │      }                                      │
  │    }                                        │
  │  }                                          │
  │                                             │
  └─── GET /api/v1/pipelines/schemas ──────────▶│
```

---

## Inference Caching Strategy

```
Frame 0   Frame 1   Frame 2   Frame 3   Frame 4   Frame 5
  │         │         │         │         │         │
  ▼         ▼         ▼         ▼         ▼         ▼

inference_interval = 3:

  [RUN]     skip      skip      [RUN]     skip      skip
    │         │         │         │         │         │
    ▼         │         │         ▼         │         │
  result A ──cache────cache    result B ──cache────cache
    │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼
  overlay   overlay   overlay   overlay   overlay   overlay
  with A    with A    with A    with B    with B    with B

Cache is also invalidated when the "feature" dropdown changes:

  Frame N: feature="detect"  →  [RUN detect]
  Frame N+1: feature="caption" (changed!) → [RUN caption] (forced)
  Frame N+2: feature="caption" →  skip (use cache)
```

---

## Local Mode vs Cloud Relay Mode

### Local Mode

```
┌──────────┐    HTTP     ┌──────────────────────────────────┐
│ Frontend │◀───────────▶│ Scope Backend (localhost:8000)   │
│ (React)  │   WebRTC    │                                  │
│          │◀═══════════▶│ PipelineRegistry                 │
└──────────┘             │   └─ "moondream" (from plugin)   │
                         │                                  │
                         │ MoondreamPipeline                │
                         │   └─ Moondream2 model on GPU     │
                         └──────────────────────────────────┘

 GET /api/v1/pipelines/schemas ──▶ local handler ──▶ JSON response
 POST /api/v1/pipeline/load    ──▶ local handler ──▶ loads model
 WebRTC stream                 ──▶ frame processor ──▶ pipeline.__call__()
```

### Cloud Relay Mode

```
┌──────────┐  WebSocket   ┌──────────────┐  httpx   ┌────────────────────────┐
│ Frontend │◀────────────▶│ fal_app.py   │◀────────▶│ Scope Backend          │
│ (React)  │              │ (WebSocket   │          │ (localhost:8000)       │
│          │              │  handler)    │  WebRTC  │                        │
│          │◀═════════════│═════════════▶│◀════════▶│ PipelineRegistry       │
└──────────┘              └──────────────┘          │  └─ "moondream"        │
                                                    │    (pre-baked in       │
                           fal.ai GPU container     │     Docker image)      │
                          ◄────────────────────────►│                        │
                                                    │ MoondreamPipeline      │
                                                    │  └─ Moondream2 on GPU  │
                                                    └────────────────────────┘

 adapter.api.getPipelineSchemas()
   └─▶ WS: {type:"api", method:"GET", path:"/api/v1/pipelines/schemas"}
         └─▶ fal_app.py handle_api_request()
               └─▶ httpx.get("http://localhost:8000/api/v1/pipelines/schemas")
                     └─▶ local handler ──▶ JSON (includes "moondream" schema)
                           └─▶ WS response ──▶ frontend renders settings UI
```

### Key Difference: Plugin Installation

```
LOCAL MODE                              CLOUD RELAY MODE
──────────                              ────────────────

  User installs plugin:                  Plugin pre-baked in Docker image:
  uv pip install -e                      Dockerfile:
    plugins/scope-moondream/               COPY cloud-plugins.txt
                                           RUN uv pip install <plugin>
  Plugin discovered at startup
  via pluggy entry points               Plugin discovered at container startup
                                         via same pluggy entry points
  Can install/uninstall at runtime
  POST /api/v1/plugins ──▶ OK           Cannot install at runtime
                                         POST /api/v1/plugins ──▶ 403 Forbidden
                                         (blocked in fal_app.py:615-622)
```

---

## Tensor Format Reference

```
┌──────────────────────────────────────────────────────────────────┐
│                       Tensor Formats                             │
│                                                                  │
│  Pipeline Input:                                                 │
│    video: List[torch.Tensor]                                     │
│    Each tensor: shape (1, H, W, C)                               │
│                   │  │  │  │                                     │
│                   │  │  │  └─ Channels: 3 (RGB)                  │
│                   │  │  └──── Width: frame width in pixels       │
│                   │  └─────── Height: frame height in pixels     │
│                   └────────── Batch: always 1                    │
│    Value range: [0, 255] uint8                                   │
│                                                                  │
│  PIL Conversion (for Moondream):                                 │
│    tensor (1,H,W,C,[0,255]) ──▶ squeeze(0) ──▶ .byte().numpy()  │
│    ──▶ Image.fromarray(arr, "RGB")                               │
│    Result: PIL Image (H, W, RGB)                                 │
│                                                                  │
│  Pipeline Output:                                                │
│    {"video": torch.Tensor}                                       │
│    Tensor: shape (1, H, W, C)                                    │
│    Value range: [0.0, 1.0] float32                               │
│                                                                  │
│  PIL → Tensor Conversion:                                        │
│    np.array(image, float32) / 255.0 ──▶ torch.from_numpy()      │
│    ──▶ .unsqueeze(0).to(device)                                  │
│                                                                  │
│  Post-pipeline (frame processor):                                │
│    output * 255.0 ──▶ .clamp(0,255) ──▶ .to(uint8)              │
│    ──▶ WebRTC encode ──▶ browser                                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Moondream Return Value Reference

```
model.caption(image, length, settings)
  └─▶ {"caption": "A person sitting at a desk working on a laptop."}

model.query(image, question, settings)
  └─▶ {"answer": "There are three people in the image."}

model.detect(image, object, settings)
  └─▶ {"objects": [
         {"x_min": 0.12, "y_min": 0.08, "x_max": 0.45, "y_max": 0.92},
         {"x_min": 0.55, "y_min": 0.15, "x_max": 0.78, "y_max": 0.85}
       ]}
       All coordinates normalized to [0, 1] relative to image dimensions.

model.point(image, object, settings)
  └─▶ {"points": [
         {"x": 0.28, "y": 0.50},
         {"x": 0.66, "y": 0.48}
       ]}
       All coordinates normalized to [0, 1] relative to image dimensions.
```
