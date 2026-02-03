# Deploying Scope to fal.ai

This guide explains how to deploy the Scope backend to fal.ai serverless.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        fal.ai Runner                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  fal_app.py                                                 │ │
│  │  ┌─────────────────┐         ┌─────────────────────────────┐│ │
│  │  │  WebSocket      │ ──────► │  Scope Backend              ││ │
│  │  │  Endpoint       │ HTTP    │  (uv run daydream-scope)    ││ │
│  │  │  /ws            │ Proxy   │  localhost:8000             ││ │
│  │  └────────┬────────┘         └──────────────┬──────────────┘│ │
│  │           │                                 │                │ │
│  └───────────┼─────────────────────────────────┼────────────────┘ │
└──────────────┼─────────────────────────────────┼─────────────────┘
               │                                 │
    WebSocket  │                      WebRTC     │
    (signaling │                      (video)    │
     + API)    │                                 │
               ▼                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                         Browser                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Frontend with FalAdapter                                   │ │
│  │  - API calls go through WebSocket                           │ │
│  │  - WebRTC signaling goes through WebSocket                  │ │
│  │  - Video frames flow directly via WebRTC                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## How It Works

1. **Single WebSocket Connection**: All communication (API calls + WebRTC signaling) goes through one WebSocket connection to prevent fal from spawning new runner instances.

2. **Scope Runs as Subprocess**: The Scope backend runs inside the fal container via `uv run daydream-scope --no-browser`.

3. **WebRTC Video Flows Directly**: Once signaling is complete, video frames flow directly via WebRTC (UDP/RTP) between browser and fal runner.

## Deployment

### 1. Deploy to fal.ai

```bash
cd scope
fal deploy fal_app.py
```

This will output a URL like: `https://fal.run/your-username/scope-app`

### 2. Update Frontend to Use FalAdapter

In your frontend initialization (e.g., `main.tsx` or `App.tsx`):

```typescript
import { initFalAdapter } from "./lib/falAdapter";

// Initialize when running on fal
const FAL_WS_URL = "wss://fal.run/your-username/scope-app/ws";

async function initApp() {
  // Check if we should use fal mode
  const useFal = import.meta.env.VITE_USE_FAL === "true";

  if (useFal) {
    const adapter = initFalAdapter(FAL_WS_URL);
    await adapter.connect();
    console.log("Connected to fal.ai backend");
  }
}

initApp();
```

### 3. Use the FalAdapter in Components

For API calls, use the adapter's API methods:

```typescript
import { getFalAdapter, isFalMode } from "./lib/falAdapter";
import { getPipelineStatus } from "./lib/api";

async function fetchStatus() {
  if (isFalMode()) {
    const adapter = getFalAdapter()!;
    return adapter.api.getPipelineStatus();
  } else {
    return getPipelineStatus();
  }
}
```

For WebRTC, use the `useWebRTCFal` hook:

```typescript
import { useWebRTC } from "./hooks/useWebRTC";
import { useWebRTCFal } from "./hooks/useWebRTCFal";
import { getFalAdapter, isFalMode } from "./lib/falAdapter";

function VideoStream() {
  // Choose the right hook based on deployment mode
  const adapter = getFalAdapter();

  const webrtc = isFalMode() && adapter
    ? useWebRTCFal({ adapter })
    : useWebRTC();

  // Use webrtc.startStream, webrtc.stopStream, etc.
}
```

## WebSocket Protocol

All messages are JSON with a `type` field.

### WebRTC Signaling

```typescript
// Get ICE servers
{ "type": "get_ice_servers" }
// Response: { "type": "ice_servers", "data": { "iceServers": [...] } }

// Send SDP offer
{ "type": "offer", "sdp": "...", "sdp_type": "offer", "initialParameters": {...} }
// Response: { "type": "answer", "sdp": "...", "sdp_type": "answer", "sessionId": "..." }

// Send ICE candidate
{ "type": "icecandidate", "sessionId": "...", "candidate": { "candidate": "...", "sdpMid": "...", "sdpMLineIndex": 0 } }
// Response: { "type": "icecandidate_ack", "status": "ok" }
```

### API Proxy

```typescript
// Make API request
{
  "type": "api",
  "method": "GET",  // or POST, PATCH, DELETE
  "path": "/api/v1/pipeline/status",
  "body": null,  // for POST/PATCH
  "request_id": "req_123"  // for correlating responses
}

// Response
{
  "type": "api_response",
  "request_id": "req_123",
  "status": 200,
  "data": { ... }
}
```

### Keepalive

```typescript
{ "type": "ping" }
// Response: { "type": "pong" }
```

## Environment Variables

The fal container inherits environment variables. Set these in your fal deployment:

- `HF_TOKEN` - Hugging Face token for TURN server access
- `PIPELINE` - Default pipeline to pre-warm (optional)

## Limitations

1. **File Downloads**: Binary file downloads (recordings, logs) need special handling. The adapter provides URLs that the browser can fetch directly.

2. **File Uploads**: Files are base64-encoded when sent through WebSocket, which increases size by ~33%.

3. **Connection Persistence**: The WebSocket connection must stay open to keep the runner alive. If it disconnects, you may get a new runner.

## Troubleshooting

### "WebSocket not connected"
Make sure `adapter.connect()` completes before making API calls.

### WebRTC connection fails
Check that TURN servers are configured. The fal runner needs a public IP or TURN relay for WebRTC to work.

### New runner spawned for each request
Make sure ALL API calls go through the FalAdapter WebSocket, not direct HTTP fetch.
