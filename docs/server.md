# Scope Server API

## Prerequisites

```bash
uv run download_models --pipeline <PIPELINE_ID>
```

Pipeline IDs: `streamdiffusionv2` (video input), `longlive` (no video input), `krea-realtime-video` (no video input), `passthrough` (testing)

## Starting the Server

```bash
uv run daydream-scope
# Custom host/port: --host 0.0.0.0 --port 8000
```

Server runs on `http://localhost:8000` by default.

## Loading a Pipeline

**Important**: Pipeline loading is **asynchronous**. The `/api/v1/pipeline/load` endpoint initiates loading in the background and returns immediately. You must poll the `/api/v1/pipeline/status` endpoint to check when the pipeline is fully loaded before starting streaming.

```javascript
// Load a pipeline (initiates async loading)
async function loadPipeline(pipelineId, loadParams = {}) {
  const response = await fetch("http://localhost:8000/api/v1/pipeline/load", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      pipeline_id: pipelineId,
      load_params: loadParams,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to load pipeline: ${error}`);
  }

  return await response.json();
}

// Check pipeline status
async function getPipelineStatus() {
  const response = await fetch("http://localhost:8000/api/v1/pipeline/status");
  return await response.json();
}

// Wait for pipeline to finish loading
async function waitForPipelineLoaded(maxWaitMs = 300000, pollIntervalMs = 1000) {
  const startTime = Date.now();

  while (Date.now() - startTime < maxWaitMs) {
    const status = await getPipelineStatus();

    if (status.status === "loaded") {
      console.log("Pipeline loaded successfully:", status);
      return status;
    } else if (status.status === "loading") {
      console.log("Pipeline still loading...");
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    } else {
      throw new Error(`Unexpected pipeline status: ${status.status}`);
    }
  }

  throw new Error("Timeout waiting for pipeline to load");
}

// Example: Load StreamDiffusionV2 pipeline and wait
await loadPipeline("streamdiffusionv2", {
  height: 512,
  width: 512,
  seed: 42,
});
await waitForPipelineLoaded();

// Example: Load LongLive pipeline and wait
await loadPipeline("longlive", {
  height: 320,
  width: 576,
  seed: 42,
});
await waitForPipelineLoaded();

// Example: Load Krea Realtime Video pipeline and wait
await loadPipeline("krea-realtime-video", {
  height: 320,
  width: 576,
  seed: 42,
  quantization: "fp8_e4m3fn", // or null for no quantization
});
await waitForPipelineLoaded();
```

### Pipeline Status Response

The `/api/v1/pipeline/status` endpoint returns:

```json
{
  "status": "loaded",
  "pipeline_id": "streamdiffusionv2",
  "load_params": {
    "height": 512,
    "width": 512,
    "seed": 42
  },
  "loaded_lora_adapters": []
}
```

**Status values**:
- `"not_loaded"` - No pipeline is loaded
- `"loading"` - Pipeline is currently being loaded
- `"loaded"` - Pipeline is ready for streaming

## Connecting to the Server

- **Video-input pipelines** (`streamdiffusionv2`): Send video to server (bidirectional)
- **No-video-input pipelines** (`longlive`, `krea-realtime-video`): Receive video only (one-way)

### For Video-Input Pipelines (e.g., streamdiffusionv2)

```javascript
// 1. Fetch ICE servers from backend (includes TURN servers for firewall traversal)
const iceServersResponse = await fetch("http://localhost:8000/api/v1/webrtc/ice-servers");
const { iceServers } = await iceServersResponse.json();

// 2. Create peer connection
const pc = new RTCPeerConnection({
  iceServers: iceServers,
});

// Store session ID for sending ICE candidates
let sessionId = null;
const queuedCandidates = [];

// 3. Create data channel
const dataChannel = pc.createDataChannel("parameters", { ordered: true });

dataChannel.onopen = () => {
  console.log("Data channel opened");
};

dataChannel.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "stream_stopped") {
    console.log("Stream stopped:", data.error_message);
    pc.close();
  }
};

// 4. Set up video element
const videoElement = document.createElement("video");
videoElement.autoplay = true;
videoElement.muted = true;
videoElement.playsInline = true;
document.body.appendChild(videoElement);

// 5. Add local video track
const localStream = await navigator.mediaDevices.getUserMedia({
  video: { width: 512, height: 512 },
});

localStream.getTracks().forEach((track) => {
  if (track.kind === "video") {
    pc.addTrack(track, localStream);
  }
});

// 6. Set up event handlers
const onTrack = (event) => {
  if (event.streams && event.streams[0]) {
    videoElement.srcObject = event.streams[0];
  }
};

const onConnectionStateChange = () => {
  console.log("Connection state:", pc.connectionState);
};

const onIceCandidate = async (event) => {
  if (event.candidate) {
    // Trickle ICE: Send candidates as they arrive
    if (sessionId) {
      await fetch(`http://localhost:8000/api/v1/webrtc/offer/${sessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candidates: [{
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
          }],
        }),
      });
    } else {
      // Queue candidates until session ID is available
      queuedCandidates.push(event.candidate);
    }
  } else {
    console.log("ICE gathering complete");
  }
};

// Attach event handlers
pc.ontrack = onTrack;
pc.onconnectionstatechange = onConnectionStateChange;
pc.onicecandidate = onIceCandidate;

// 7. Create offer and send immediately (Trickle ICE)
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const sdpResponse = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    initialParameters: {
      prompts: [{ text: "A beautiful landscape", weight: 1.0 }],
      prompt_interpolation_method: "linear",
      denoising_step_list: [700, 500],
    },
  }),
});

const answerData = await sdpResponse.json();
sessionId = answerData.sessionId; // Store session ID

await pc.setRemoteDescription({
  type: answerData.type,
  sdp: answerData.sdp,
});

// Flush queued candidates
if (queuedCandidates.length > 0) {
  await fetch(`http://localhost:8000/api/v1/webrtc/offer/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidates: queuedCandidates.map(c => ({
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      })),
    }),
  });
  queuedCandidates.length = 0;
}
```

### For No-Video-Input Pipelines (e.g., longlive, krea-realtime-video)

```javascript
// 1. Fetch ICE servers from backend (includes TURN servers for firewall traversal)
const iceServersResponse = await fetch("http://localhost:8000/api/v1/webrtc/ice-servers");
const { iceServers } = await iceServersResponse.json();

// 2. Create peer connection
const pc = new RTCPeerConnection({
  iceServers: iceServers,
});

// Store session ID for sending ICE candidates
let sessionId = null;
const queuedCandidates = [];

// 3. Create data channel
const dataChannel = pc.createDataChannel("parameters", { ordered: true });

dataChannel.onopen = () => {
  console.log("Data channel opened");
  // Send initial parameters
  dataChannel.send(
    JSON.stringify({
      prompts: [{ text: "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.", weight: 100 }],
      prompt_interpolation_method: "linear",
      denoising_step_list: [1000, 750, 500, 250],
      manage_cache: true,
    })
  );
};

dataChannel.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "stream_stopped") {
    console.log("Stream stopped:", data.error_message);
    pc.close();
  }
};

// 4. Add video transceiver (after data channel, before event handlers)
pc.addTransceiver("video");

// 5. Set up video element
const videoElement = document.createElement("video");
videoElement.autoplay = true;
videoElement.muted = true;
videoElement.playsInline = true;
document.body.appendChild(videoElement);

// 6. Set up event handlers
const onTrack = (event) => {
  if (event.streams && event.streams[0]) {
    videoElement.srcObject = event.streams[0];
  }
};

const onConnectionStateChange = () => {
  console.log("Connection state:", pc.connectionState);
};

const onIceConnectionStateChange = () => {
  console.log("ICE connection state:", pc.iceConnectionState);
};

const onIceCandidate = async (event) => {
  if (event.candidate) {
    // Trickle ICE: Send candidates as they arrive
    if (sessionId) {
      await fetch(`http://localhost:8000/api/v1/webrtc/offer/${sessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candidates: [{
            candidate: event.candidate.candidate,
            sdpMid: event.candidate.sdpMid,
            sdpMLineIndex: event.candidate.sdpMLineIndex,
          }],
        }),
      });
    } else {
      // Queue candidates until session ID is available
      queuedCandidates.push(event.candidate);
    }
  } else {
    console.log("ICE gathering complete");
  }
};

// Attach event handlers
pc.ontrack = onTrack;
pc.onconnectionstatechange = onConnectionStateChange;
pc.oniceconnectionstatechange = onIceConnectionStateChange;
pc.onicecandidate = onIceCandidate;

// 7. Create offer and send immediately (Trickle ICE)
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const sdpResponse = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    initialParameters: {
      prompts: [{ text: "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.", weight: 100 }],
      prompt_interpolation_method: "linear",
      denoising_step_list: [1000, 750, 500, 250],
      manage_cache: true,
    },
  }),
});

const answerData = await sdpResponse.json();
sessionId = answerData.sessionId; // Store session ID

await pc.setRemoteDescription({
  type: answerData.type,
  sdp: answerData.sdp,
});

// Flush queued candidates
if (queuedCandidates.length > 0) {
  await fetch(`http://localhost:8000/api/v1/webrtc/offer/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidates: queuedCandidates.map(c => ({
        candidate: c.candidate,
        sdpMid: c.sdpMid,
        sdpMLineIndex: c.sdpMLineIndex,
      })),
    }),
  });
  queuedCandidates.length = 0;
}
```

- **Order:** Create data channel → Add transceiver/track → Set event handlers → Create offer
- **Video element:** Use `autoplay`, `muted`, `playsInline` attributes

## React/Component Framework Integration

```javascript
const [remoteStream, setRemoteStream] = useState(null);
const videoRef = useRef(null);

pc.ontrack = (event) => {
  if (event.streams && event.streams[0]) {
    setRemoteStream(event.streams[0]);
  }
};

useEffect(() => {
  if (videoRef.current && remoteStream) {
    videoRef.current.srcObject = remoteStream;
  } else if (videoRef.current && !remoteStream) {
    videoRef.current.srcObject = null;
  }
}, [remoteStream]);
```

```jsx
<video ref={videoRef} autoPlay muted playsInline />
```

## Sending Parameters

```javascript
const dataChannel = pc.createDataChannel("parameters", { ordered: true });

dataChannel.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "stream_stopped") {
    pc.close();
  }
};

// Send parameter updates
dataChannel.send(JSON.stringify({
  prompts: [{ text: "A cat", weight: 1.0 }],
  prompt_interpolation_method: "slerp",
  denoising_step_list: [600, 400],
  noise_scale: 0.8, // StreamDiffusionV2
  paused: false,
  reset_cache: true, // when manage_cache is false
}));
```

## Parameters

- **`prompts`** (array): `[{ text: string, weight: number }]` - Prompts with weights for spatial blending
- **`prompt_interpolation_method`** (string): `"linear"` or `"slerp"` (default: `"linear"`)
- **`transition`** (object): `{ target_prompts: [...], num_steps: number, temporal_interpolation_method: "linear"|"slerp" }` - Smooth prompt transitions
- **`denoising_step_list`** (array): Descending timesteps (e.g., `[700, 500]`)
- **`noise_scale`** (number, StreamDiffusionV2): `0.0-1.0` - Noise amount
- **`noise_controller`** (boolean, StreamDiffusionV2): Auto-adjust noise scale
- **`manage_cache`** (boolean, krea-realtime-video/longlive): Auto cache management
- **`reset_cache`** (boolean): Reset cache (when `manage_cache` is false)
- **`kv_cache_attention_bias`** (number, krea-realtime-video): `0.01-1.0` - Past frame reliance
- **`paused`** (boolean): Pause/resume generation


## API Endpoints

- `POST /api/v1/pipeline/load` - Load a pipeline
- `GET /api/v1/pipeline/status` - Get pipeline status
- `GET /api/v1/webrtc/ice-servers` - Get ICE server configuration (includes TURN servers)
- `POST /api/v1/webrtc/offer` - Establish WebRTC connection (returns `sessionId`)
- `PATCH /api/v1/webrtc/offer/{session_id}` - Send ICE candidate(s) (Trickle ICE)
- `GET /api/v1/models/status?pipeline_id=<ID>` - Check if models are downloaded
- `POST /api/v1/models/download` - Download models for a pipeline
- `GET /api/v1/hardware/info` - Get hardware information
- `GET /health` - Health check endpoint
- `GET /docs` - API documentation (Swagger UI)
