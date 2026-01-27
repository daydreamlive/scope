# Plan: Move fal.ai Integration from Frontend to Server

## Overview

Move the fal serverless SDK integration from the frontend to the scope server, enabling:
1. Local input sources (webcam via WebRTC, Spout) to be sent to fal cloud for GPU inference
2. Inference results returned to scope server
3. Output sent via Spout to other applications

## Reference Implementation Analysis

**Source:** [fal-ai-community/fal-demos/yolo_webcam_webrtc](https://github.com/fal-ai-community/fal-demos/tree/main/fal_demos/video/yolo_webcam_webrtc)

### Key Patterns from Reference

1. **fal.ai acts as WebRTC server** - accepts offers, sends answers
2. **Client creates offers** - browser (or Scope server) initiates WebRTC connection
3. **Simple signaling protocol:**
   - Server sends `{"type": "ready"}` when WebSocket connects
   - Client sends `{"type": "offer", "sdp": "..."}`
   - Server responds `{"type": "answer", "sdp": "..."}`
   - Both exchange `{"type": "icecandidate", "candidate": {...}}`

4. **Token authentication:**
   ```
   POST https://rest.alpha.fal.ai/tokens/
   Authorization: Key {api_key}
   Body: {"allowed_apps": [alias], "token_expiration": 120}
   ```
   WebSocket URL: `wss://fal.run/{appId}?fal_jwt_token={token}`

5. **Track processing pattern (YOLOTrack):**
   ```python
   @pc.on("track")
   def on_track(track):
       if track.kind == "video":
           pc.addTrack(create_processing_track(track, model))
   ```
   The server wraps incoming track with a processing track that transforms each frame.

### Differences from Our Use Case

| Aspect | Reference (YOLO) | Our Use Case |
|--------|------------------|--------------|
| **Who is client?** | Browser | Scope Server |
| **Who is server?** | fal.ai | fal.ai (same) |
| **Input source** | Browser webcam | Spout/WebRTC from Scope |
| **Processing** | YOLO detection | Video diffusion pipeline |
| **Output destination** | Browser video element | Spout sender |

**Key insight:** Since fal.ai always acts as WebRTC server, Scope server must act as WebRTC client (create offers, receive answers). This is the *opposite* of how Scope server handles browser WebRTC connections.

---

## Current Architecture (Frontend-based fal)

```
Browser ──WebRTC──► fal.ai ──proxy──► Scope Backend ──► GPU ──► Scope Backend ──► fal.ai ──WebRTC──► Browser
                   (WebSocket)
```

**Current Implementation:**
- `fal_app.py`: fal serverless app that spawns Scope backend as subprocess
- `falAdapter.ts`: WebSocket client for API proxying + WebRTC signaling
- `falContext.tsx`: React context provider for fal mode
- `useUnifiedWebRTC.ts`: Mode-agnostic WebRTC hook

## Proposed Architecture (Server-based fal)

```
Local Input ──► Scope Server ──WebRTC Client──► fal.ai ──► GPU Inference ──► fal.ai ──WebRTC──► Scope Server ──► Spout Output
(Spout/WebRTC)                 (WebSocket)
```

**Key Change:** Scope server becomes a WebRTC *client* to fal.ai instead of the browser being the client.

---

## Implementation Plan

### Phase 1: Create Server-Side fal Client Module

**New file: `src/scope/server/fal_client.py`**

This module handles WebSocket and WebRTC connection to fal.ai from the server. Based on the reference implementation, Scope acts as the WebRTC *client* (creates offers).

```python
import asyncio
import json
import logging
from typing import Callable

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from av import VideoFrame

logger = logging.getLogger(__name__)

TOKEN_EXPIRATION_SECONDS = 120


class FalClient:
    """WebSocket + WebRTC client for connecting to fal.ai cloud.

    Based on fal-demos/yolo_webcam_webrtc reference implementation.
    Scope acts as WebRTC client (creates offers), fal.ai acts as server.
    """

    def __init__(
        self,
        app_id: str,
        api_key: str,
        on_frame_received: Callable[[VideoFrame], None] | None = None,
    ):
        self.app_id = app_id  # e.g., "owner/app-name/webrtc"
        self.api_key = api_key
        self.on_frame_received = on_frame_received

        self.ws: websockets.WebSocketClientProtocol | None = None
        self.pc: RTCPeerConnection | None = None
        self.output_track: "FalOutputTrack | None" = None
        self.stop_event = asyncio.Event()
        self._receive_task: asyncio.Task | None = None

    async def _get_temporary_token(self) -> str:
        """Get temporary JWT token from fal API (mirrors frontend pattern)."""
        import aiohttp

        # Extract alias from app_id (e.g., "owner/app-name/webrtc" -> "app-name")
        parts = self.app_id.split("/")
        alias = parts[1] if len(parts) >= 2 else self.app_id

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://rest.alpha.fal.ai/tokens/",
                headers={
                    "Authorization": f"Key {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "allowed_apps": [alias],
                    "token_expiration": TOKEN_EXPIRATION_SECONDS,
                },
            ) as resp:
                if not resp.ok:
                    error_body = await resp.text()
                    raise RuntimeError(f"Token request failed: {resp.status} {error_body}")
                token = await resp.json()
                # Handle both string and object responses
                if isinstance(token, dict) and "detail" in token:
                    return token["detail"]
                return token

    def _build_ws_url(self, token: str) -> str:
        """Build WebSocket URL with JWT token (mirrors frontend pattern)."""
        app_id = self.app_id.strip("/")
        return f"wss://fal.run/{app_id}?fal_jwt_token={token}"

    async def connect(self) -> None:
        """Connect to fal WebSocket and establish WebRTC connection."""
        # Get temporary token
        token = await self._get_temporary_token()
        ws_url = self._build_ws_url(token)

        logger.info(f"Connecting to fal WebSocket: {ws_url[:50]}...")
        self.ws = await websockets.connect(ws_url)

        # Wait for "ready" message from server
        ready_msg = await self.ws.recv()
        ready_data = json.loads(ready_msg)
        if ready_data.get("type") != "ready":
            raise RuntimeError(f"Expected 'ready' message, got: {ready_data}")
        logger.info("fal server ready")

        # Create peer connection
        self.pc = RTCPeerConnection(
            configuration={"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}
        )

        # Set up event handlers
        self._setup_pc_handlers()

        # Add output track (for sending frames to fal)
        from scope.server.fal_tracks import FalOutputTrack
        self.output_track = FalOutputTrack()
        self.pc.addTrack(self.output_track)

        # Create and send offer (we are the client)
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        await self.ws.send(json.dumps({
            "type": "offer",
            "sdp": self.pc.localDescription.sdp,
        }))
        logger.info("Sent WebRTC offer")

        # Start message receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

    def _setup_pc_handlers(self):
        """Set up RTCPeerConnection event handlers."""

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if self.ws is None:
                return
            if candidate is None:
                await self.ws.send(json.dumps({
                    "type": "icecandidate",
                    "candidate": None,
                }))
            else:
                await self.ws.send(json.dumps({
                    "type": "icecandidate",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex,
                    },
                }))

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {self.pc.connectionState}")
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                self.stop_event.set()

        @self.pc.on("track")
        def on_track(track):
            """Handle incoming track (processed frames from fal)."""
            if track.kind == "video":
                logger.info("Received video track from fal")
                asyncio.create_task(self._consume_track(track))

    async def _consume_track(self, track):
        """Consume frames from the incoming track."""
        while not self.stop_event.is_set():
            try:
                frame = await track.recv()
                if self.on_frame_received:
                    self.on_frame_received(frame)
            except Exception as e:
                logger.error(f"Error receiving frame: {e}")
                break

    async def _receive_loop(self):
        """Receive and handle WebSocket messages."""
        try:
            while not self.stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message: {message}")
                    continue

                msg_type = data.get("type")

                if msg_type == "answer":
                    # Set remote description from server's answer
                    answer = RTCSessionDescription(
                        sdp=data["sdp"],
                        type="answer",
                    )
                    await self.pc.setRemoteDescription(answer)
                    logger.info("Set remote description from answer")

                elif msg_type == "icecandidate":
                    candidate_data = data.get("candidate")
                    if candidate_data is None:
                        await self.pc.addIceCandidate(None)
                    else:
                        candidate = candidate_from_sdp(candidate_data.get("candidate", ""))
                        candidate.sdpMid = candidate_data.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                        await self.pc.addIceCandidate(candidate)

                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('error')}")

                else:
                    logger.debug(f"Unknown message type: {msg_type}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Receive loop error: {e}")
        finally:
            self.stop_event.set()

    async def send_frame(self, frame: VideoFrame) -> None:
        """Send a frame to fal for processing."""
        if self.output_track:
            await self.output_track.put_frame(frame)

    async def disconnect(self) -> None:
        """Close WebRTC and WebSocket connections."""
        self.stop_event.set()

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self.pc:
            await self.pc.close()
            self.pc = None

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("Disconnected from fal")
```

### Phase 2: Create fal Video Track for Sending Frames

**New file: `src/scope/server/fal_tracks.py`**

Custom aiortc MediaStreamTrack for sending frames to fal. This follows the same pattern as `YOLOTrack` in the reference, but for outbound frames.

```python
import asyncio
import fractions
import time

from aiortc.mediastreams import MediaStreamTrack
from av import VideoFrame


class FalOutputTrack(MediaStreamTrack):
    """Sends frames from queue to fal via WebRTC.

    This is the outbound track - frames are put into the queue
    and sent to fal.ai for processing.
    """

    kind = "video"

    def __init__(self, target_fps: int = 30):
        super().__init__()
        self.frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)
        self.target_fps = target_fps
        self._start_time = time.time()
        self._frame_count = 0

    async def recv(self) -> VideoFrame:
        """Called by aiortc to get next frame to send.

        This method is called by the WebRTC stack when it needs
        the next frame to encode and send.
        """
        frame = await self.frame_queue.get()

        # Set pts (presentation timestamp) and time_base
        self._frame_count += 1
        frame.pts = self._frame_count
        frame.time_base = fractions.Fraction(1, self.target_fps)

        return frame

    async def put_frame(self, frame: VideoFrame) -> bool:
        """Add frame to be sent to fal.

        Returns True if frame was queued, False if queue was full (frame dropped).
        """
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            # Drop oldest frame and add new one
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                return True
            except asyncio.QueueEmpty:
                return False

    def put_frame_sync(self, frame: VideoFrame) -> bool:
        """Synchronous version for use from non-async contexts."""
        return self.put_frame_nowait(frame)

    def put_frame_nowait(self, frame: VideoFrame) -> bool:
        """Non-blocking frame put."""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            return False


class FalInputTrack(MediaStreamTrack):
    """Receives processed frames from fal via WebRTC.

    This wraps an incoming track and makes frames available via a queue.
    Similar pattern to YOLOTrack in reference, but stores frames instead
    of processing them.
    """

    kind = "video"

    def __init__(self, source_track: MediaStreamTrack):
        super().__init__()
        self.source_track = source_track
        self.frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)
        self._consume_task: asyncio.Task | None = None

    def start_consuming(self):
        """Start consuming frames from source track."""
        self._consume_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self):
        """Continuously receive frames from source and queue them."""
        while True:
            try:
                frame = await self.source_track.recv()
                try:
                    self.frame_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    # Drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except asyncio.QueueEmpty:
                        pass
            except Exception:
                break

    async def recv(self) -> VideoFrame:
        """Get next received frame."""
        return await self.frame_queue.get()

    def get_frame_nowait(self) -> VideoFrame | None:
        """Non-blocking frame get."""
        try:
            return self.frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def stop(self):
        """Stop consuming frames."""
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
```

### Phase 3: Integrate fal Client with FrameProcessor

**Modify: `src/scope/server/frame_processor.py`**

Add fal cloud processing mode alongside existing local pipeline processing:

```python
class FrameProcessor:
    def __init__(self, ...):
        # Existing attributes...

        # fal cloud integration
        self.fal_client: FalClient | None = None
        self.fal_enabled = False
        self._fal_received_frames: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=30)

    def _on_fal_frame_received(self, frame: VideoFrame):
        """Callback when frame is received from fal."""
        try:
            self._fal_received_frames.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest
            try:
                self._fal_received_frames.get_nowait()
                self._fal_received_frames.put_nowait(frame)
            except asyncio.QueueEmpty:
                pass

    async def connect_to_fal(self, app_id: str, api_key: str) -> None:
        """Connect to fal.ai cloud for remote processing."""
        if self.fal_client:
            await self.fal_client.disconnect()

        self.fal_client = FalClient(
            app_id=app_id,
            api_key=api_key,
            on_frame_received=self._on_fal_frame_received,
        )
        await self.fal_client.connect()
        self.fal_enabled = True

    async def disconnect_from_fal(self) -> None:
        """Disconnect from fal.ai cloud."""
        if self.fal_client:
            await self.fal_client.disconnect()
            self.fal_client = None
        self.fal_enabled = False

    def put(self, frame: VideoFrame) -> bool:
        """Put frame for processing."""
        if self.fal_enabled and self.fal_client:
            # Send to fal cloud via WebRTC
            return self.fal_client.output_track.put_frame_nowait(frame)
        else:
            # Existing local processing
            ...

    def get(self) -> VideoFrame | None:
        """Get processed frame."""
        if self.fal_enabled:
            # Get from fal cloud
            try:
                return self._fal_received_frames.get_nowait()
            except asyncio.QueueEmpty:
                return None
        else:
            # Existing local processing
            ...
```

### Phase 4: Add API Endpoints for fal Configuration

**Modify: `src/scope/server/app.py`**

Add REST endpoints to configure fal mode:

```python
from pydantic import BaseModel


class FalConnectRequest(BaseModel):
    app_id: str  # e.g., "owner/scope-fal/webrtc"
    api_key: str


class FalStatusResponse(BaseModel):
    connected: bool
    app_id: str | None = None


@app.post("/api/v1/fal/connect")
async def connect_to_fal(request: FalConnectRequest) -> FalStatusResponse:
    """Connect to fal.ai cloud for remote GPU inference."""
    await frame_processor.connect_to_fal(
        app_id=request.app_id,
        api_key=request.api_key,
    )
    return FalStatusResponse(connected=True, app_id=request.app_id)


@app.post("/api/v1/fal/disconnect")
async def disconnect_from_fal() -> FalStatusResponse:
    """Disconnect from fal.ai cloud."""
    await frame_processor.disconnect_from_fal()
    return FalStatusResponse(connected=False)


@app.get("/api/v1/fal/status")
async def get_fal_status() -> FalStatusResponse:
    """Get current fal connection status."""
    if frame_processor.fal_enabled and frame_processor.fal_client:
        return FalStatusResponse(
            connected=True,
            app_id=frame_processor.fal_client.app_id,
        )
    return FalStatusResponse(connected=False)
```

### Phase 5: Handle Spout Input → fal → Spout Output Flow

The complete data flow with Spout:

```
Spout Receiver → FrameProcessor.put() → FalOutputTrack → WebRTC → fal.ai GPU
                                                                        │
Spout Sender ← FrameProcessor.get() ← _fal_received_frames ← WebRTC ←───┘
```

The existing `_spout_receiver_loop` and `_spout_sender_loop` already handle async frame I/O. The fal integration slots in at the FrameProcessor level transparently.

### Phase 6: Parameter Forwarding and UI Integration

#### Current Parameter Flow (Local Mode)

```
Browser UI ─── WebRTC Data Channel ──► Scope Server ──► FrameProcessor.update_parameters()
                  (JSON messages)                              │
                                                               ▼
                                                        Pipeline Processors
```

**Key insight:** Parameters are sent via WebRTC data channel as JSON messages, NOT via HTTP/REST. This includes:
- `prompts`, `noise_scale`, `denoising_step_list`
- `kv_cache_attention_bias`, `paused`
- `spout_sender`, `spout_receiver`
- `vace_ref_images`, `vace_context_scale`
- `transition` (prompt interpolation)
- `ctrl_input` (controller input)
- `lora_scales`

#### Required: Parameter Forwarding to fal Cloud

When cloud mode is enabled, the FalClient must forward parameter updates to fal.ai via its own data channel:

```
Browser UI ─── WebRTC Data Channel ──► Scope Server ──► FalClient Data Channel ──► fal.ai
                  (JSON messages)           │                (JSON messages)
                                            │
                                            ▼
                                    Also stored locally
                                    (for UI state sync)
```

#### FalClient Data Channel Implementation

Add data channel support to `FalClient`:

```python
class FalClient:
    def __init__(self, ...):
        # ... existing attributes ...
        self.data_channel: RTCDataChannel | None = None
        self._pending_parameters: dict = {}

    async def connect(self, initial_parameters: dict | None = None) -> None:
        """Connect to fal with optional initial parameters."""
        # ... token and WebSocket setup ...

        # Create peer connection
        self.pc = RTCPeerConnection(...)

        # Create data channel for parameter updates (BEFORE creating offer)
        self.data_channel = self.pc.createDataChannel(
            "parameters",
            ordered=True,  # Ensure parameter order is preserved
        )

        @self.data_channel.on("open")
        def on_data_channel_open():
            logger.info("Data channel to fal opened")
            # Send any pending parameters
            if self._pending_parameters:
                self._send_parameters(self._pending_parameters)
                self._pending_parameters = {}

        # ... rest of connection setup ...

        # Include initial parameters in offer message
        await self.ws.send(json.dumps({
            "type": "offer",
            "sdp": self.pc.localDescription.sdp,
            "initialParameters": initial_parameters,  # Sent with offer
        }))

    def send_parameters(self, parameters: dict) -> bool:
        """Forward parameter update to fal.ai via data channel."""
        if self.data_channel and self.data_channel.readyState == "open":
            return self._send_parameters(parameters)
        else:
            # Queue for when channel opens
            self._pending_parameters.update(parameters)
            return False

    def _send_parameters(self, parameters: dict) -> bool:
        """Internal: send parameters over data channel."""
        try:
            # Filter out None values (same as frontend)
            filtered = {k: v for k, v in parameters.items() if v is not None}
            message = json.dumps(filtered)
            self.data_channel.send(message)
            logger.debug(f"Sent parameters to fal: {filtered}")
            return True
        except Exception as e:
            logger.error(f"Failed to send parameters: {e}")
            return False
```

#### FrameProcessor Parameter Routing

Modify `update_parameters()` to route to fal when cloud mode is active:

```python
def update_parameters(self, parameters: dict[str, Any]):
    """Update parameters - routes to local pipelines OR fal cloud."""

    # Handle Spout config locally (always)
    if "spout_sender" in parameters:
        self._update_spout_sender(parameters.pop("spout_sender"))
    if "spout_receiver" in parameters:
        self._update_spout_receiver(parameters.pop("spout_receiver"))

    # Route remaining parameters based on mode
    if self.fal_enabled and self.fal_client:
        # Forward to fal cloud
        self.fal_client.send_parameters(parameters)
    else:
        # Local processing
        for processor in self.pipeline_processors:
            processor.update_parameters(parameters)

    # Always store locally for state tracking
    self.parameters = {**self.parameters, **parameters}
```

#### Parameters That Stay Local vs Forward to fal

| Parameter | Local | Forward to fal | Notes |
|-----------|-------|----------------|-------|
| `spout_sender` | ✓ | ✗ | Output is always local |
| `spout_receiver` | ✓ | ✗ | Input is always local |
| `paused` | ✓ | ✓ | Both need to know |
| `prompts` | ✗ | ✓ | Pipeline parameter |
| `noise_scale` | ✗ | ✓ | Pipeline parameter |
| `denoising_step_list` | ✗ | ✓ | Pipeline parameter |
| `kv_cache_attention_bias` | ✗ | ✓ | Pipeline parameter |
| `transition` | ✗ | ✓ | Pipeline parameter |
| `vace_*` | ✗ | ✓ | Pipeline parameter |
| `ctrl_input` | ✗ | ✓ | Pipeline parameter |
| `lora_scales` | ✗ | ✓ | Pipeline parameter |

#### UI Toggle: Cloud vs Local Mode

Design goals:
1. **Single UI toggle** to switch between local GPU and fal cloud
2. **Seamless switching** - same parameter controls work in both modes
3. **Clear status indication** - user knows which mode is active
4. **Graceful fallback** - if cloud fails, can switch back to local

#### Frontend Changes

**New State in Settings Context**

Modify `frontend/src/context/SettingsContext.tsx`:

```typescript
interface Settings {
  // ... existing settings ...

  // Cloud inference settings
  cloudMode: {
    enabled: boolean;
    appId: string;      // e.g., "username/scope-fal/webrtc"
    apiKey: string;     // fal API key
    status: "disconnected" | "connecting" | "connected" | "error";
    errorMessage?: string;
  };
}

const defaultSettings: Settings = {
  // ... existing defaults ...
  cloudMode: {
    enabled: false,
    appId: "",
    apiKey: "",
    status: "disconnected",
  },
};
```

**Cloud Mode Toggle Component**

New file: `frontend/src/components/CloudModeToggle.tsx`

```typescript
export function CloudModeToggle() {
  const { settings, updateSettings } = useSettings();
  const { cloudMode } = settings;

  const handleToggle = async (enabled: boolean) => {
    if (enabled) {
      // Connect to fal cloud
      updateSettings({ cloudMode: { ...cloudMode, status: "connecting" } });
      try {
        await fetch("/api/v1/fal/connect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            app_id: cloudMode.appId,
            api_key: cloudMode.apiKey,
            initial_parameters: {
              pipeline_ids: settings.pipelineIds,
              prompts: settings.prompts,
              // ... other current parameters
            },
          }),
        });
        updateSettings({ cloudMode: { ...cloudMode, enabled: true, status: "connected" } });
      } catch (error) {
        updateSettings({
          cloudMode: {
            ...cloudMode,
            enabled: false,
            status: "error",
            errorMessage: error.message,
          },
        });
      }
    } else {
      // Disconnect from fal cloud
      await fetch("/api/v1/fal/disconnect", { method: "POST" });
      updateSettings({ cloudMode: { ...cloudMode, enabled: false, status: "disconnected" } });
    }
  };

  return (
    <div className="cloud-mode-toggle">
      <Switch
        checked={cloudMode.enabled}
        onCheckedChange={handleToggle}
        disabled={cloudMode.status === "connecting"}
      />
      <span>
        {cloudMode.enabled ? "☁️ Cloud GPU" : "💻 Local GPU"}
      </span>
      {cloudMode.status === "connecting" && <Spinner />}
      {cloudMode.status === "error" && (
        <span className="error">{cloudMode.errorMessage}</span>
      )}
    </div>
  );
}
```

**Settings Panel for Cloud Credentials**

Modify `frontend/src/components/SettingsPanel.tsx` to add a section for cloud configuration:

```typescript
<Section title="Cloud Inference">
  <TextInput
    label="fal App ID"
    value={settings.cloudMode.appId}
    onChange={(appId) => updateSettings({
      cloudMode: { ...settings.cloudMode, appId }
    })}
    placeholder="username/scope-fal/webrtc"
  />
  <TextInput
    label="fal API Key"
    type="password"
    value={settings.cloudMode.apiKey}
    onChange={(apiKey) => updateSettings({
      cloudMode: { ...settings.cloudMode, apiKey }
    })}
    placeholder="Enter your fal API key"
  />
  <CloudModeToggle />
</Section>
```

#### Backend API Changes

Modify `src/scope/server/app.py` to update connect endpoint to accept initial parameters:

```python
class FalConnectRequest(BaseModel):
    app_id: str
    api_key: str
    initial_parameters: dict | None = None  # Pipeline params at connect time


@app.post("/api/v1/fal/connect")
async def connect_to_fal(request: FalConnectRequest) -> FalStatusResponse:
    """Connect to fal.ai cloud for remote GPU inference."""
    await frame_processor.connect_to_fal(
        app_id=request.app_id,
        api_key=request.api_key,
        initial_parameters=request.initial_parameters,
    )
    return FalStatusResponse(connected=True, app_id=request.app_id)
```

#### Data Flow with Cloud Mode Toggle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  Frontend UI                                     │
│                                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │ Cloud Toggle    │    │              Parameter Controls                   │   │
│  │ [OFF] Local GPU │    │  Prompts | Noise | Steps | VACE | LoRA | etc     │   │
│  │ [ON]  Cloud GPU │    │                                                   │   │
│  └────────┬────────┘    └──────────────────────┬───────────────────────────┘   │
│           │                                     │                               │
│           │ POST /api/v1/fal/connect           │ WebRTC Data Channel           │
│           │ POST /api/v1/fal/disconnect        │ (same as before)              │
│           ▼                                     ▼                               │
└───────────┼─────────────────────────────────────┼───────────────────────────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              Scope Server                                      │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         FrameProcessor                                   │ │
│  │                                                                         │ │
│  │   fal_enabled: bool ◄─── Set by /api/v1/fal/connect                    │ │
│  │                                                                         │ │
│  │   update_parameters(params):                                            │ │
│  │       if fal_enabled:                                                   │ │
│  │           fal_client.send_parameters(params)  ──► To fal cloud          │ │
│  │       else:                                                             │ │
│  │           pipeline_processors.update(params)  ──► Local processing      │ │
│  │                                                                         │ │
│  │   put(frame):                                                           │ │
│  │       if fal_enabled:                                                   │ │
│  │           fal_client.send_frame(frame)        ──► To fal cloud          │ │
│  │       else:                                                             │ │
│  │           local_queue.put(frame)              ──► Local processing      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

#### Persistence

Store cloud credentials in localStorage (frontend) so users don't have to re-enter:

```typescript
// In SettingsContext
useEffect(() => {
  const saved = localStorage.getItem("cloudModeSettings");
  if (saved) {
    const { appId, apiKey } = JSON.parse(saved);
    updateSettings({ cloudMode: { ...settings.cloudMode, appId, apiKey } });
  }
}, []);

useEffect(() => {
  // Don't persist the enabled state, only credentials
  localStorage.setItem("cloudModeSettings", JSON.stringify({
    appId: settings.cloudMode.appId,
    apiKey: settings.cloudMode.apiKey,
  }));
}, [settings.cloudMode.appId, settings.cloudMode.apiKey]);
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/scope/server/fal_client.py` | WebSocket + WebRTC client for fal.ai (with data channel) |
| `src/scope/server/fal_tracks.py` | Custom MediaStreamTrack classes for frame I/O |
| `frontend/src/components/CloudModeToggle.tsx` | UI toggle for cloud/local mode |

## Files to Modify

| File | Changes |
|------|---------|
| `src/scope/server/frame_processor.py` | Add fal cloud mode, parameter routing, connect/disconnect logic |
| `src/scope/server/app.py` | Add fal configuration endpoints |
| `src/scope/server/schema.py` | Add fal configuration schemas |
| `frontend/src/context/SettingsContext.tsx` | Add cloudMode state |
| `frontend/src/components/SettingsPanel.tsx` | Add cloud credentials UI |
| `pyproject.toml` | Add `aiohttp` dependency for token API |

---

## Configuration

### API Usage

```bash
# Connect to fal
curl -X POST http://localhost:8000/api/v1/fal/connect \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "your-username/scope-fal/webrtc",
    "api_key": "your-fal-api-key"
  }'

# Check status
curl http://localhost:8000/api/v1/fal/status

# Disconnect
curl -X POST http://localhost:8000/api/v1/fal/disconnect
```

### Environment Variables

```bash
# Optional: Set default fal credentials
FAL_APP_ID=your-username/scope-fal/webrtc
FAL_API_KEY=your-key
```

---

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    "aiohttp>=3.9.0",  # For token API requests
    # websockets and aiortc already included
]
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Scope Server                                    │
│                                                                             │
│  ┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐          │
│  │ Spout        │────►│ FrameProcessor  │────►│ FalClient       │          │
│  │ Receiver     │     │                 │     │                 │          │
│  └──────────────┘     │  - put()        │     │ - WebSocket     │          │
│                       │  - get()        │     │ - RTCPeerConn   │          │
│  ┌──────────────┐     │  - fal_enabled  │     │ - FalOutputTrack│          │
│  │ Spout        │◄────│                 │◄────│                 │          │
│  │ Sender       │     └─────────────────┘     └────────┬────────┘          │
│  └──────────────┘                                      │                    │
│                                                        │ WebRTC             │
│  ┌──────────────┐                                      │ (client mode)      │
│  │ Browser      │◄─── WebRTC (server mode) ◄───────────┤                    │
│  │ Preview      │                                      │                    │
│  └──────────────┘                                      │                    │
└────────────────────────────────────────────────────────┼────────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │      fal.ai Cloud    │
                                              │                      │
                                              │  WebRTC Endpoint     │
                                              │  (/webrtc)           │
                                              │         │            │
                                              │         ▼            │
                                              │  ┌────────────────┐  │
                                              │  │ Scope Pipeline │  │
                                              │  │ (GPU Inference)│  │
                                              │  └────────────────┘  │
                                              │                      │
                                              └──────────────────────┘
```

---

## Signaling Protocol (from Reference)

```
┌─────────────┐                           ┌─────────────┐
│ Scope Server│                           │   fal.ai    │
│  (Client)   │                           │  (Server)   │
└──────┬──────┘                           └──────┬──────┘
       │                                         │
       │──── WebSocket Connect ─────────────────►│
       │                                         │
       │◄──── {"type": "ready"} ─────────────────│
       │                                         │
       │──── {"type": "offer", "sdp": "..."} ───►│
       │                                         │
       │◄──── {"type": "answer", "sdp": "..."} ──│
       │                                         │
       │◄───► ICE Candidates (bidirectional) ───►│
       │                                         │
       │═════ WebRTC Media Stream ══════════════►│
       │                                         │
       │◄════ WebRTC Media Stream (processed) ═══│
       │                                         │
```

---

## Phase-by-Phase Testing

This section provides verification steps for each phase. Complete all tests for a phase before proceeding to the next.

### Phase 1 Testing: FalClient Module

**Prerequisites:** None (first phase)

#### Unit Tests

Create `tests/server/test_fal_client.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_get_temporary_token_success():
    """Test successful token acquisition from fal API."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="test-key")

    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.json = AsyncMock(return_value={"detail": "test-token"})
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        token = await client._get_temporary_token()
        assert token == "test-token"

@pytest.mark.asyncio
async def test_get_temporary_token_failure():
    """Test token acquisition failure handling."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="bad-key")

    with patch("aiohttp.ClientSession") as mock_session:
        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(RuntimeError, match="Token request failed"):
            await client._get_temporary_token()

def test_build_ws_url():
    """Test WebSocket URL construction."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="owner/app-name/webrtc", api_key="test-key")
    url = client._build_ws_url("my-token")
    assert url == "wss://fal.run/owner/app-name/webrtc?fal_jwt_token=my-token"

def test_build_ws_url_strips_slashes():
    """Test URL construction handles leading/trailing slashes."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="/owner/app-name/webrtc/", api_key="test-key")
    url = client._build_ws_url("my-token")
    assert url == "wss://fal.run/owner/app-name/webrtc?fal_jwt_token=my-token"
```

Run with:
```bash
uv run pytest tests/server/test_fal_client.py -v
```

#### Manual Tests

1. **Token API Test** (requires real fal API key):
   ```bash
   # Test token endpoint directly
   curl -X POST https://rest.alpha.fal.ai/tokens/ \
     -H "Authorization: Key $FAL_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"allowed_apps": ["scope-fal"], "token_expiration": 120}'
   ```
   Expected: Returns a JWT token string

2. **Import Test**:
   ```bash
   uv run python -c "from scope.server.fal_client import FalClient; print('FalClient imported successfully')"
   ```
   Expected: No import errors

#### Phase 1 Completion Criteria
- [ ] All unit tests pass
- [ ] Token API returns valid token with real API key
- [ ] FalClient module imports without errors

---

### Phase 2 Testing: FalOutputTrack and FalInputTrack

**Prerequisites:** Phase 1 complete

#### Unit Tests

Create `tests/server/test_fal_tracks.py`:

```python
import pytest
import asyncio
from av import VideoFrame
import numpy as np

@pytest.mark.asyncio
async def test_fal_output_track_put_and_recv():
    """Test frame queue put and receive."""
    from scope.server.fal_tracks import FalOutputTrack

    track = FalOutputTrack(target_fps=30)

    # Create test frame
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = VideoFrame.from_ndarray(arr, format="rgb24")

    # Put frame
    result = await track.put_frame(frame)
    assert result is True

    # Receive frame
    received = await track.recv()
    assert received.pts == 1
    assert received.time_base.numerator == 1
    assert received.time_base.denominator == 30

@pytest.mark.asyncio
async def test_fal_output_track_queue_full_drops_oldest():
    """Test that full queue drops oldest frame."""
    from scope.server.fal_tracks import FalOutputTrack

    track = FalOutputTrack(target_fps=30)
    track.frame_queue = asyncio.Queue(maxsize=2)  # Small queue for testing

    arr = np.zeros((480, 640, 3), dtype=np.uint8)

    # Fill queue
    for i in range(3):
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        await track.put_frame(frame)

    # Queue should have frames 1 and 2 (0 was dropped)
    assert track.frame_queue.qsize() == 2

def test_fal_output_track_put_frame_nowait():
    """Test non-blocking frame put."""
    from scope.server.fal_tracks import FalOutputTrack

    track = FalOutputTrack(target_fps=30)

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = VideoFrame.from_ndarray(arr, format="rgb24")

    result = track.put_frame_nowait(frame)
    assert result is True
    assert track.frame_queue.qsize() == 1
```

Run with:
```bash
uv run pytest tests/server/test_fal_tracks.py -v
```

#### Manual Tests

1. **Track Creation Test**:
   ```bash
   uv run python -c "
   from scope.server.fal_tracks import FalOutputTrack, FalInputTrack
   track = FalOutputTrack()
   print(f'Track kind: {track.kind}')
   print(f'Queue maxsize: {track.frame_queue.maxsize}')
   print('FalOutputTrack created successfully')
   "
   ```
   Expected: `Track kind: video`, queue maxsize 30

2. **Frame Round-Trip Test**:
   ```bash
   uv run python -c "
   import asyncio
   import numpy as np
   from av import VideoFrame
   from scope.server.fal_tracks import FalOutputTrack

   async def test():
       track = FalOutputTrack()
       arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
       frame = VideoFrame.from_ndarray(arr, format='rgb24')
       await track.put_frame(frame)
       received = await track.recv()
       print(f'Frame received: {received.width}x{received.height}, pts={received.pts}')

   asyncio.run(test())
   "
   ```
   Expected: `Frame received: 640x480, pts=1`

#### Phase 2 Completion Criteria
- [ ] All unit tests pass
- [ ] FalOutputTrack can queue and retrieve frames
- [ ] Frame timestamps are correctly set

---

### Phase 3 Testing: FrameProcessor Integration

**Prerequisites:** Phases 1-2 complete

#### Unit Tests

Add to `tests/server/test_frame_processor.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_connect_to_fal():
    """Test fal connection initialization."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)  # Use appropriate constructor

    with patch("scope.server.fal_client.FalClient") as MockFalClient:
        mock_client = AsyncMock()
        MockFalClient.return_value = mock_client

        await processor.connect_to_fal(
            app_id="owner/app/webrtc",
            api_key="test-key"
        )

        assert processor.fal_enabled is True
        assert processor.fal_client is not None
        mock_client.connect.assert_called_once()

@pytest.mark.asyncio
async def test_disconnect_from_fal():
    """Test fal disconnection cleanup."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)
    processor.fal_client = AsyncMock()
    processor.fal_enabled = True

    await processor.disconnect_from_fal()

    assert processor.fal_enabled is False
    assert processor.fal_client is None

def test_put_routes_to_fal_when_enabled():
    """Test frame routing to fal when cloud mode enabled."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)
    processor.fal_enabled = True
    processor.fal_client = MagicMock()
    processor.fal_client.output_track = MagicMock()
    processor.fal_client.output_track.put_frame_nowait = MagicMock(return_value=True)

    frame = MagicMock()
    result = processor.put(frame)

    processor.fal_client.output_track.put_frame_nowait.assert_called_once_with(frame)
```

Run with:
```bash
uv run pytest tests/server/test_frame_processor.py -v -k fal
```

#### Manual Tests

1. **Server Startup Test** (no fal connection):
   ```bash
   uv run daydream-scope
   # Server should start without errors
   # fal_enabled should be False by default
   ```

2. **FrameProcessor State Test**:
   ```bash
   uv run python -c "
   from scope.server.frame_processor import FrameProcessor
   # Check that fal attributes exist
   import inspect
   source = inspect.getsource(FrameProcessor.__init__)
   assert 'fal_client' in source or hasattr(FrameProcessor, 'fal_client')
   print('FrameProcessor has fal integration attributes')
   "
   ```

#### Phase 3 Completion Criteria
- [ ] All unit tests pass
- [ ] Server starts without errors
- [ ] FrameProcessor has fal_client and fal_enabled attributes
- [ ] Local processing still works (no regression)

---

### Phase 4 Testing: API Endpoints

**Prerequisites:** Phases 1-3 complete

#### Unit Tests

Add to `tests/server/test_app.py`:

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

def test_fal_connect_endpoint():
    """Test /api/v1/fal/connect endpoint."""
    from scope.server.app import app

    with patch("scope.server.app.frame_processor") as mock_fp:
        mock_fp.connect_to_fal = AsyncMock()

        client = TestClient(app)
        response = client.post(
            "/api/v1/fal/connect",
            json={"app_id": "owner/app/webrtc", "api_key": "test-key"}
        )

        assert response.status_code == 200
        assert response.json()["connected"] is True
        assert response.json()["app_id"] == "owner/app/webrtc"

def test_fal_disconnect_endpoint():
    """Test /api/v1/fal/disconnect endpoint."""
    from scope.server.app import app

    with patch("scope.server.app.frame_processor") as mock_fp:
        mock_fp.disconnect_from_fal = AsyncMock()

        client = TestClient(app)
        response = client.post("/api/v1/fal/disconnect")

        assert response.status_code == 200
        assert response.json()["connected"] is False

def test_fal_status_endpoint_disconnected():
    """Test /api/v1/fal/status when disconnected."""
    from scope.server.app import app

    with patch("scope.server.app.frame_processor") as mock_fp:
        mock_fp.fal_enabled = False
        mock_fp.fal_client = None

        client = TestClient(app)
        response = client.get("/api/v1/fal/status")

        assert response.status_code == 200
        assert response.json()["connected"] is False

def test_fal_status_endpoint_connected():
    """Test /api/v1/fal/status when connected."""
    from scope.server.app import app

    with patch("scope.server.app.frame_processor") as mock_fp:
        mock_fp.fal_enabled = True
        mock_fp.fal_client.app_id = "owner/app/webrtc"

        client = TestClient(app)
        response = client.get("/api/v1/fal/status")

        assert response.status_code == 200
        assert response.json()["connected"] is True
        assert response.json()["app_id"] == "owner/app/webrtc"
```

Run with:
```bash
uv run pytest tests/server/test_app.py -v -k fal
```

#### Manual Tests

1. **API Endpoint Test** (server must be running):
   ```bash
   # Start server in one terminal
   uv run daydream-scope

   # In another terminal, test endpoints

   # Test status (should be disconnected)
   curl http://localhost:8000/api/v1/fal/status
   # Expected: {"connected": false, "app_id": null}

   # Test connect (will fail without valid credentials, but endpoint should respond)
   curl -X POST http://localhost:8000/api/v1/fal/connect \
     -H "Content-Type: application/json" \
     -d '{"app_id": "test/app/webrtc", "api_key": "invalid"}'
   # Expected: Error response (token fetch fails)

   # Test disconnect
   curl -X POST http://localhost:8000/api/v1/fal/disconnect
   # Expected: {"connected": false, "app_id": null}
   ```

2. **Schema Validation Test**:
   ```bash
   # Test invalid request body
   curl -X POST http://localhost:8000/api/v1/fal/connect \
     -H "Content-Type: application/json" \
     -d '{"invalid": "data"}'
   # Expected: 422 Validation Error
   ```

#### Phase 4 Completion Criteria
- [ ] All unit tests pass
- [ ] /api/v1/fal/status returns correct disconnected state
- [ ] /api/v1/fal/connect validates request body
- [ ] /api/v1/fal/disconnect returns success

---

### Phase 5 Testing: Spout Integration

**Prerequisites:** Phases 1-4 complete, fal app deployed

#### Unit Tests

```python
@pytest.mark.asyncio
async def test_fal_frame_callback_queues_frame():
    """Test that received frames are queued."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)

    frame = MagicMock()
    processor._on_fal_frame_received(frame)

    assert processor._fal_received_frames.qsize() == 1

def test_get_returns_fal_frame_when_enabled():
    """Test that get() returns frames from fal queue."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)
    processor.fal_enabled = True

    frame = MagicMock()
    processor._fal_received_frames.put_nowait(frame)

    result = processor.get()
    assert result is frame
```

#### Manual Tests

1. **End-to-End with Real fal** (requires deployed fal app):
   ```bash
   # Deploy fal app if not already
   fal deploy fal_app.py

   # Start scope server
   uv run daydream-scope

   # Connect to fal
   curl -X POST http://localhost:8000/api/v1/fal/connect \
     -H "Content-Type: application/json" \
     -d "{\"app_id\": \"$FAL_APP_ID\", \"api_key\": \"$FAL_API_KEY\"}"

   # Check connection status
   curl http://localhost:8000/api/v1/fal/status
   # Expected: {"connected": true, "app_id": "..."}
   ```

2. **Spout Flow Test** (Windows only, requires Spout-compatible apps):
   ```
   1. Start a Spout sender app (e.g., OBS with Spout plugin)
   2. Start scope server with Spout receiver enabled
   3. Connect to fal via API
   4. Start a Spout receiver app (e.g., Resolume)
   5. Verify video flows through the entire pipeline
   ```

3. **WebRTC Connection Verification**:
   ```bash
   # Check server logs for:
   # - "Connecting to fal WebSocket..."
   # - "fal server ready"
   # - "Sent WebRTC offer"
   # - "Set remote description from answer"
   # - "Connection state: connected"
   # - "Received video track from fal"
   ```

#### Phase 5 Completion Criteria
- [ ] All unit tests pass
- [ ] Can connect to deployed fal app via API
- [ ] Server logs show successful WebRTC connection
- [ ] (If Spout available) Frames flow through full pipeline

---

### Phase 6 Testing: Parameter Forwarding and UI

**Prerequisites:** Phases 1-5 complete

#### Unit Tests

```python
def test_send_parameters_queues_when_channel_closed():
    """Test parameters are queued when data channel not ready."""
    from scope.server.fal_client import FalClient

    client = FalClient(app_id="test", api_key="test")
    client.data_channel = None  # Not connected

    result = client.send_parameters({"prompt": "test"})

    assert result is False
    assert client._pending_parameters == {"prompt": "test"}

def test_update_parameters_routes_to_fal():
    """Test parameter routing when fal enabled."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)
    processor.fal_enabled = True
    processor.fal_client = MagicMock()
    processor.fal_client.send_parameters = MagicMock(return_value=True)

    processor.update_parameters({"prompts": ["test prompt"]})

    processor.fal_client.send_parameters.assert_called_once()

def test_spout_params_stay_local():
    """Test Spout parameters are not forwarded to fal."""
    from scope.server.frame_processor import FrameProcessor

    processor = FrameProcessor(...)
    processor.fal_enabled = True
    processor.fal_client = MagicMock()
    processor._update_spout_sender = MagicMock()

    processor.update_parameters({
        "spout_sender": {"enabled": True},
        "prompts": ["test"]
    })

    # Spout handled locally
    processor._update_spout_sender.assert_called_once()
    # Only prompts sent to fal
    call_args = processor.fal_client.send_parameters.call_args[0][0]
    assert "spout_sender" not in call_args
```

#### Manual Tests

1. **Parameter Forwarding Test** (requires connected fal):
   ```bash
   # With fal connected, open browser to scope UI
   # Change prompt in UI
   # Check fal logs for received parameter update
   ```

2. **Data Channel Test**:
   ```bash
   # Check server logs for:
   # - "Data channel to fal opened"
   # - "Sent parameters to fal: {...}"
   ```

3. **UI Toggle Test** (requires frontend changes):
   ```
   1. Open scope UI in browser
   2. Enter fal credentials in settings
   3. Toggle cloud mode ON
   4. Verify status shows "connected"
   5. Change parameters (prompt, noise, etc.)
   6. Check fal logs for parameter updates
   7. Toggle cloud mode OFF
   8. Verify local processing resumes
   ```

4. **Persistence Test**:
   ```
   1. Enter fal credentials in UI
   2. Refresh browser page
   3. Open settings panel
   4. Verify credentials are still filled in
   5. Verify cloud mode is OFF (not auto-connected)
   ```

#### Phase 6 Completion Criteria
- [ ] All unit tests pass
- [ ] Parameters are forwarded to fal when connected
- [ ] Spout parameters stay local
- [ ] UI toggle connects/disconnects correctly
- [ ] Credentials persist across page refresh
- [ ] Mode switching works without errors

---

## Test Summary Checklist

Use this checklist to track progress through all phases:

| Phase | Unit Tests | Manual Tests | Status |
|-------|------------|--------------|--------|
| 1. FalClient Module | 3 tests | 2 tests | ⬜ |
| 2. FalOutputTrack/FalInputTrack | 3 tests | 2 tests | ⬜ |
| 3. FrameProcessor Integration | 3 tests | 2 tests | ⬜ |
| 4. API Endpoints | 4 tests | 2 tests | ⬜ |
| 5. Spout Integration | 2 tests | 3 tests | ⬜ |
| 6. Parameter Forwarding & UI | 3 tests | 4 tests | ⬜ |

**Total: 18 unit tests, 15 manual tests**

---

## Considerations

### Thread Safety
- FalClient runs in asyncio event loop
- Spout threads communicate via queues
- Use `asyncio.run_coroutine_threadsafe()` for cross-thread async calls

### Error Handling
- WebSocket disconnection: Auto-reconnect with exponential backoff
- WebRTC ICE failures: Log and notify, allow manual retry
- Frame timeouts: Drop frames and log warnings
- Token expiration: Re-authenticate before 120s timeout

### Latency
- WebRTC adds ~50-100ms latency per direction
- Total round-trip to fal cloud: ~200-400ms depending on network
- Consider frame rate adjustment based on measured latency

### Fallback
- If fal connection fails, option to fall back to local processing (if GPU available)
- Configuration flag: `fal_cloud.fallback_to_local: true`
