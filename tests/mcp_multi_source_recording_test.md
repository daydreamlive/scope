# Multi-Source/Multi-Sink Recording Test (MCP)

This test verifies the full multi-source/multi-sink headless pipeline flow
via MCP tools, including per-sink frame capture and per-node recording.

## Prerequisites

- Two test videos: `frontend/public/assets/test.mp4` and `frontend/public/assets/test2.mp4`
- Pipeline `passthrough` loaded (or any pipeline that passes frames through)
- Scope running on a known port (e.g. 8000)

## Test Steps

### 1. Connect to Scope

```
connect_to_scope(port=<PORT>)
```

### 2. Load the passthrough pipeline

```
load_pipeline(pipeline_id="passthrough")
```

### 3. Start stream with multi-source/multi-sink graph

Start a headless stream with two video file sources, two passthrough pipelines,
two sinks, and one record node:

```
start_stream(graph={
  "nodes": [
    {"id": "source_1", "type": "source", "source_mode": "video_file", "source_name": "<absolute_path>/test.mp4"},
    {"id": "source_2", "type": "source", "source_mode": "video_file", "source_name": "<absolute_path>/test2.mp4"},
    {"id": "pipeline_1", "type": "pipeline", "pipeline_id": "passthrough"},
    {"id": "pipeline_2", "type": "pipeline", "pipeline_id": "passthrough"},
    {"id": "output_1", "type": "sink"},
    {"id": "output_2", "type": "sink"},
    {"id": "record_1", "type": "record"}
  ],
  "edges": [
    {"from": "source_1", "from_port": "video", "to_node": "pipeline_1", "to_port": "video", "kind": "stream"},
    {"from": "source_2", "from_port": "video", "to_node": "pipeline_2", "to_port": "video", "kind": "stream"},
    {"from": "pipeline_1", "from_port": "video", "to_node": "output_1", "to_port": "video", "kind": "stream"},
    {"from": "pipeline_2", "from_port": "video", "to_node": "output_2", "to_port": "video", "kind": "stream"},
    {"from": "pipeline_1", "from_port": "video", "to_node": "record_1", "to_port": "video", "kind": "stream"}
  ]
}, input_mode="video")
```

Expected: `status: ok`, `sink_node_ids: ["output_1", "output_2"]`, `source_node_ids: ["source_1", "source_2"]`

### 4. Wait for frames

Wait 3-5 seconds for the pipeline to produce frames.

### 5. Capture frame from sink 1

```
capture_frame(sink_node_id="output_1")
```

Expected: Returns a valid JPEG image file path. Read the file to verify it's a valid image.

### 6. Capture frame from sink 2

```
capture_frame(sink_node_id="output_2")
```

Expected: Returns a valid JPEG image file path. Read the file to verify it's a valid image.

### 7. Start recording

```
start_recording()
```

Expected: `status: started`

### 8. Wait for recording

Wait 3-5 seconds to accumulate recording frames.

### 9. Stop recording

```
stop_recording()
```

Expected: Returns `status: stopped` with a file path.

### 10. Download recording

```
download_recording()
```

Expected: Returns an MP4 file path. Verify the file exists and has non-zero size.

### 11. Stop stream

```
stop_stream()
```

Expected: `status: ok`

## Cloud Mode Variant

To run this test in cloud mode, first start two Scope instances:

```bash
# Terminal 1 — "cloud" instance:
SCOPE_CLOUD_WS=1 uv run daydream-scope --port 8002

# Terminal 2 — "local" instance:
SCOPE_CLOUD_WS_URL=ws://localhost:8002/ws SCOPE_CLOUD_APP_ID=local uv run daydream-scope --port 8022
```

Then:
1. `connect_to_scope(port=8022)`
2. `connect_to_cloud()` — waits for cloud WebSocket connection
3. `load_pipeline(pipeline_id="passthrough")` — loaded on cloud side
4. Follow steps 3-11 above (stream runs through cloud relay)
5. `disconnect_from_cloud()`
