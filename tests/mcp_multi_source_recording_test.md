# MCP Multi-Source Recording Test

Manual test scenario for verifying headless multi-source streaming with recording via the MCP server.

## Prerequisites

- Scope server running (`uv run daydream-scope`)
- Split-screen plugin installed (`scope-split-screen`)
- Test video files available at `frontend/public/assets/test.mp4` and `frontend/public/assets/test2.mp4`

## Sample MCP Prompt

Use the following prompt with Claude Code (connected to the Scope MCP server) to run the test:

```
Connect to Scope on port 8000, then:

1. Load the split-screen pipeline.
2. Start a headless stream with this graph:
   - Source Node 1 (video_file): frontend/public/assets/test.mp4
   - Source Node 2 (video_file): frontend/public/assets/test2.mp4
   - Pipeline Node: split-screen (source_1 → "video" input, source_2 → "video2" input)
   - Sink Node: attached to the pipeline "video" output
3. Wait a few seconds for frames to flow, then capture a frame from the sink. Verify the image is a valid JPEG showing a split-screen (top half from test.mp4, bottom half from test2.mp4).
4. Start recording.
5. Wait ~5 seconds, then stop recording.
6. Download the recording. Verify it is a valid MP4 video (H.264, correct resolution, multiple frames).
7. Stop the stream.
```

## Graph Configuration

```json
{
  "nodes": [
    {"id": "source_1", "type": "source", "source_mode": "video_file", "source_name": "<absolute-path>/frontend/public/assets/test.mp4"},
    {"id": "source_2", "type": "source", "source_mode": "video_file", "source_name": "<absolute-path>/frontend/public/assets/test2.mp4"},
    {"id": "pipeline_1", "type": "pipeline", "pipeline_id": "split-screen"},
    {"id": "output_1", "type": "sink"}
  ],
  "edges": [
    {"from": "source_1", "from_port": "video", "to_node": "pipeline_1", "to_port": "video", "kind": "stream"},
    {"from": "source_2", "from_port": "video", "to_node": "pipeline_1", "to_port": "video2", "kind": "stream"},
    {"from": "pipeline_1", "from_port": "video", "to_node": "output_1", "to_port": "video", "kind": "stream"}
  ]
}
```

## MCP Tools Used

| Step | Tool | Key Arguments |
|------|------|---------------|
| Connect | `connect_to_scope` | `port=8000` |
| Load | `load_pipeline` | `pipeline_id="split-screen"` |
| Start | `start_stream` | `input_mode="video"`, `graph={...}` |
| Capture | `capture_frame` | `sink_node_id="output_1"` |
| Record | `start_recording` | (none) |
| Stop rec | `stop_recording` | (none) |
| Download | `download_recording` | (none) |
| Stop | `stop_stream` | (none) |

## Expected Results

| Check | Expected |
|-------|----------|
| `start_stream` response | `status: "ok"`, `sink_node_ids: ["output_1"]`, `source_node_ids: ["source_1", "source_2"]` |
| Captured frame | Valid JPEG, 512x512, split-screen content visible |
| `start_recording` response | `status: "started"` |
| `stop_recording` response | `status: "stopped"`, `file_path` present |
| Downloaded recording | Valid MP4, H.264, 512x512, 30fps, ~5s duration |
| Server logs | No errors |

## REST API Endpoints (used by MCP tools)

```
POST /api/v1/session/start          - Start headless stream
GET  /api/v1/session/frame           - Capture frame
POST /api/v1/session/recording/start - Start recording
POST /api/v1/session/recording/stop  - Stop recording
GET  /api/v1/session/recording/download - Download MP4
POST /api/v1/session/stop            - Stop stream
```
