# Generate Endpoint

Batch video generation via HTTP. Unlike the WebRTC streaming path (real-time, interactive), the generate endpoint produces a complete video in one request, processing it chunk-by-chunk with SSE progress events.

Primary consumer: ComfyUI custom nodes (`comfyui-scope`).

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/generate` | POST | Generate video (SSE stream) |
| `/api/v1/generate/cancel` | POST | Cancel after current chunk |
| `/api/v1/generate/upload` | POST | Upload input video for v2v |
| `/api/v1/generate/upload-data` | POST | Upload binary data blob (VACE, per-chunk video) |
| `/api/v1/generate/download` | GET | Download output video |

Only one generation can run at a time (409 if busy).

## Flow

```
1. [optional] POST /generate/upload      → input_path
2. [optional] POST /generate/upload-data  → data_blob_path
3. POST /generate (JSON body, references paths from steps 1-2)
   ← SSE: event: progress  {chunk, total_chunks, frames, latency, fps}
   ← SSE: event: complete  {output_path, video_shape, num_frames, ...}
4. GET /generate/download?path=<output_path>
   ← binary video data
```

## Binary Protocol

### Video Upload (`/generate/upload`)

**Request**: Raw uint8 bytes in THWC order (frames × height × width × channels).

**Headers** (required):
- `X-Video-Frames`: T
- `X-Video-Height`: H
- `X-Video-Width`: W
- `X-Video-Channels`: C (default 3)

**Stored format**: 20-byte header + raw data.
```
[4 bytes: ndim (little-endian u32)]
[4 bytes × ndim: shape dimensions (little-endian u32 each)]
[raw uint8 video bytes]
```

### Data Blob Upload (`/generate/upload-data`)

**Request**: Raw binary blob containing packed arrays. Max size: 2 GB.

The blob is an opaque byte buffer. `ChunkSpec` entries in the generate request reference regions of this blob by offset:

```json
{
  "chunk": 0,
  "vace_frames_offset": 0,
  "vace_frames_shape": [1, 3, 12, 320, 576],
  "vace_masks_offset": 26542080,
  "vace_masks_shape": [1, 1, 12, 320, 576]
}
```

Arrays are packed as contiguous float32 (VACE frames/masks) or uint8 (input video). The client is responsible for computing offsets when packing the blob.

### Video Download (`/generate/download`)

**Response**: Same binary format as upload (20-byte header + raw uint8 THWC data).

**Response headers**:
- `X-Video-Frames`, `X-Video-Height`, `X-Video-Width`, `X-Video-Channels`

## GenerateRequest

```json
{
  "pipeline_id": "longlive",
  "prompt": "a cat walking",
  "num_frames": 48,
  "seed": 42,
  "noise_scale": 0.7,
  "input_path": "<from /generate/upload>",
  "data_blob_path": "<from /generate/upload-data>",
  "chunk_specs": [
    {
      "chunk": 0,
      "text": "override prompt for chunk 0",
      "lora_scales": {"path/to/lora.safetensors": 0.5},
      "vace_frames_offset": 0,
      "vace_frames_shape": [1, 3, 12, 320, 576]
    }
  ],
  "pre_processor_id": null,
  "post_processor_id": null
}
```

Request-level fields are global defaults. `chunk_specs` entries override any field for a specific chunk index. Only fields that change need to be specified — prompts are sticky (last-set persists).

See `schema.py` for the full `GenerateRequest` and `ChunkSpec` field definitions.
