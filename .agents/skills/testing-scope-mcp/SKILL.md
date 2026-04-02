---
name: testing-scope-mcp
description: Test Daydream Scope through its stdio MCP server, including disconnected startup, `connect_to_scope`, direct Python stdio client fallback, and lightweight smoke tests. Use when testing Scope via MCP, debugging MCP registration, or when a workflow depends on `connect_to_scope`, `load_pipeline`, `start_stream`, or `capture_frame`.
---

# Testing Scope MCP

## Quick Start

Use this skill for a reusable MCP workflow against a running Scope server.

Default posture:
- Start the MCP server with `uv run daydream-scope --mcp`
- Connect explicitly with `connect_to_scope(port=...)`
- If Cursor does not surface the `scope` MCP tools, use the direct stdio client fallback below
- Use a lightweight pipeline such as `gray` for the first smoke test

## Standard Workflow

1. Start or identify the target Scope HTTP server and note its port.
2. Start the Scope MCP server over stdio:

```bash
UV_NO_SYNC=1 uv run daydream-scope --mcp
```

3. Connect MCP to the running Scope instance (replace `8022` with the actual port):

```python
await session.call_tool("connect_to_scope", {"port": 8022})
```

## Workspace MCP Registration

To have Cursor expose the `scope` MCP tools automatically, verify `.mcp.json` contains:

```json
{
  "mcpServers": {
    "scope": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "daydream-scope", "--mcp"]
    }
  }
}
```

If the tools are still missing after reload/reconnect, use the direct stdio fallback instead of blocking on Cursor setup.

## Direct Stdio MCP Fallback

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command="uv",
        args=["run", "daydream-scope", "--mcp"],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("connect_to_scope", {"port": 8022})
            tools = await session.list_tools()
            print([tool.name for tool in tools.tools])


asyncio.run(main())
```

Replace `8022` with the actual Scope HTTP port. If this client can list tools, continue testing through it even if Cursor has not exposed the tool descriptors.

## Smoke Test

Before testing a larger workflow, run a minimal smoke test against the `gray` pipeline (no model downloads required).

1. Create a test video:

```bash
ffmpeg -y -f lavfi -i "testsrc2=size=512x512:rate=8:duration=4" -pix_fmt yuv420p /tmp/scope-mcp-gray-input.mp4
```

2. MCP sequence:
- `connect_to_scope(port=8022)`
- `load_pipeline(pipeline_id="gray")`
- `start_stream(pipeline_id="gray", input_mode="video", input_source={"enabled": true, "source_type": "video_file", "source_name": "/tmp/scope-mcp-gray-input.mp4"})`
- `capture_frame()`
- `stop_stream()`

3. Verify from evidence, not just status codes:
- Pipeline loads successfully
- `capture_frame` returns an image path
- The captured image is actually grayscale, not merely non-empty

## Key Rules

- `start_stream` takes `pipeline_id` only, not a graph
- `capture_frame` returns a file path — read it back for verification
- If the MCP server was started disconnected, `connect_to_scope(...)` must happen first
- If MCP is unavailable, fall back to the direct stdio client or the HTTP APIs below

## HTTP API Fallback

When MCP tools are unavailable or the MCP server cannot be started, use these endpoints directly against the Scope HTTP server:

| Operation | Endpoint |
|-----------|----------|
| Connect to cloud | `POST /api/v1/cloud/connect` body: `{"app_id": "..."}` |
| Cloud status | `GET /api/v1/cloud/status` |
| Resolve workflow | `POST /api/v1/workflow/resolve` body: `{"pipelines": [...]}` |
| Load pipeline | `POST /api/v1/pipeline/load` body: `{"pipeline_ids": ["name"]}` |
| Pipeline status | `GET /api/v1/pipeline/status` |
| Start session | `POST /api/v1/session/start` body: `{"pipeline_id": "name", ...}` |
| Capture frame | `GET /api/v1/session/frame` (returns JPEG binary) |
| Stop session | `POST /api/v1/session/stop` |
| Start recording | `POST /api/v1/recordings/headless/start` |
| Stop recording | `POST /api/v1/recordings/headless/stop` |
| Download recording | `GET /api/v1/recordings/headless` (returns MP4 binary) |
| Tail logs | `GET /api/v1/logs/tail?lines=30` |

## Troubleshooting

When something fails, isolate the boundary first:
1. MCP server startup
2. `connect_to_scope`
3. Tool discovery
4. Pipeline load
5. Session start
6. Frame capture

Read logs from the failing process, make the smallest viable fix, and rerun the failing step before rerunning the full workflow.
