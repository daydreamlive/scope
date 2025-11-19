# Scope Server API

## Prerequisites

```bash
uv sync
uv run build --server-only
uv run download_models --pipeline <PIPELINE_ID>
```

Pipeline IDs: `streamdiffusionv2` (video input), `longlive` (no video input), `krea-realtime-video` (no video input), `passthrough` (testing)

## Starting the Server

```bash
uv run daydream-scope --server-only
# Custom host/port: --host 0.0.0.0 --port 8000
```

Server runs on `http://localhost:8000` by default.

## Loading a Pipeline

```javascript
// Load a pipeline
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

// Example: Load StreamDiffusionV2 pipeline
await loadPipeline("streamdiffusionv2", {
  height: 512,
  width: 512,
  seed: 42,
});

// Example: Load LongLive pipeline
await loadPipeline("longlive", {
  height: 320,
  width: 576,
  seed: 42,
});

// Example: Load Krea Realtime Video pipeline
await loadPipeline("krea-realtime-video", {
  height: 320,
  width: 576,
  seed: 42,
  quantization: "fp8_e4m3fn", // or null for no quantization
});
```

```javascript
async function getPipelineStatus() {
  const response = await fetch("http://localhost:8000/api/v1/pipeline/status");
  return await response.json();
}
```

## Connecting to the Server

- **Video-input pipelines** (`streamdiffusionv2`): Send video to server (bidirectional)
- **No-video-input pipelines** (`longlive`, `krea-realtime-video`): Receive video only (one-way)

### For Video-Input Pipelines (e.g., streamdiffusionv2)

```javascript
// 1. Create peer connection
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
});

// 2. Create data channel
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

// 3. Set up video element
const videoElement = document.createElement("video");
videoElement.autoplay = true;
videoElement.muted = true;
videoElement.playsInline = true;
document.body.appendChild(videoElement);

// 4. Add local video track
const localStream = await navigator.mediaDevices.getUserMedia({
  video: { width: 512, height: 512 },
});

localStream.getTracks().forEach((track) => {
  if (track.kind === "video") {
    pc.addTrack(track, localStream);
  }
});

// 5. Set up event handlers
const onTrack = (event) => {
  if (event.streams && event.streams[0]) {
    videoElement.srcObject = event.streams[0];
  }
};

const onConnectionStateChange = () => {
  console.log("Connection state:", pc.connectionState);
};

const onIceCandidate = async (event) => {
  if (event.candidate === null) {
    // ICE gathering complete
    const sdpResponse = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
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

    const answer = {
      type: "answer",
      sdp: (await sdpResponse.json()).sdp,
    };

    await pc.setRemoteDescription(answer);
  }
};

// Attach event handlers
pc.ontrack = onTrack;
pc.onconnectionstatechange = onConnectionStateChange;
pc.onicecandidate = onIceCandidate;

// 6. Create offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
```

### For No-Video-Input Pipelines (e.g., longlive, krea-realtime-video)

```javascript
// 1. Create peer connection
const pc = new RTCPeerConnection({
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
});

// 2. Create data channel
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

// 3. Add video transceiver (after data channel, before event handlers)
pc.addTransceiver("video");

// 4. Set up video element
const videoElement = document.createElement("video");
videoElement.autoplay = true;
videoElement.muted = true;
videoElement.playsInline = true;
document.body.appendChild(videoElement);

// 5. Set up event handlers
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
    console.log("ICE candidate:", event.candidate);
  } else {
    // ICE gathering complete
    console.log("ICE gathering complete, sending offer to server");
    const sdpResponse = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
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
    const answer = {
      type: "answer",
      sdp: answerData.sdp,
    };

    await pc.setRemoteDescription(answer);
    console.log("WebRTC connection established");
  }
};

// Attach event handlers
pc.ontrack = onTrack;
pc.onconnectionstatechange = onConnectionStateChange;
pc.oniceconnectionstatechange = onIceConnectionStateChange;
pc.onicecandidate = onIceCandidate;

// 6. Create offer
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
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
- `POST /api/v1/webrtc/offer` - Establish WebRTC connection
- `GET /api/v1/models/status?pipeline_id=<ID>` - Check if models are downloaded
- `POST /api/v1/models/download` - Download models for a pipeline
- `GET /api/v1/hardware/info` - Get hardware information
- `GET /health` - Health check endpoint
- `GET /docs` - API documentation (Swagger UI)
