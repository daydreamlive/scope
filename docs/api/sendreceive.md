# Send and Receive Video

This guide shows how to set up a WebRTC connection for **video-to-video mode** - sending input video (webcam, screen, or file) and receiving generated video.

## Overview

In video-to-video mode:

- You send video frames as input (e.g., webcam, screen capture)
- The server transforms the video based on your prompts
- Generated video is streamed back in real-time

## Prerequisites

1. Server is running: `uv run daydream-scope`
2. Models are downloaded for your pipeline
3. Pipeline is loaded (see [Load Pipeline](load.md))

## Complete Example

```javascript
async function startBidirectionalStream(inputStream, initialPrompt = "A painting") {
  const API_BASE = "http://localhost:8000";

  // 1. Get ICE servers
  const iceResponse = await fetch(`${API_BASE}/api/v1/webrtc/ice-servers`);
  const { iceServers } = await iceResponse.json();

  // 2. Create peer connection
  const pc = new RTCPeerConnection({ iceServers });

  // State
  let sessionId = null;
  const queuedCandidates = [];

  // 3. Create data channel
  const dataChannel = pc.createDataChannel("parameters", { ordered: true });

  dataChannel.onopen = () => {
    console.log("Data channel ready");
  };

  dataChannel.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "stream_stopped") {
      console.log("Stream stopped:", data.error_message);
      pc.close();
    }
  };

  // 4. Add LOCAL video track (for sending to server)
  inputStream.getTracks().forEach((track) => {
    if (track.kind === "video") {
      console.log("Adding video track for sending");
      pc.addTrack(track, inputStream);
    }
  });

  // 5. Handle REMOTE video track (from server)
  pc.ontrack = (event) => {
    if (event.streams && event.streams[0]) {
      document.getElementById("outputVideo").srcObject = event.streams[0];
    }
  };

  // 6. Connection monitoring
  pc.onconnectionstatechange = () => {
    console.log("Connection state:", pc.connectionState);
  };

  // 7. ICE candidate handling
  pc.onicecandidate = async (event) => {
    if (event.candidate) {
      if (sessionId) {
        await sendIceCandidate(sessionId, event.candidate);
      } else {
        queuedCandidates.push(event.candidate);
      }
    }
  };

  // 8. Create and send offer
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const response = await fetch(`${API_BASE}/api/v1/webrtc/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      initialParameters: {
        input_mode: "video",
        prompts: [{ text: initialPrompt, weight: 1.0 }],
        denoising_step_list: [700, 500]
      }
    })
  });

  const answer = await response.json();
  sessionId = answer.sessionId;

  // 9. Set remote description
  await pc.setRemoteDescription({
    type: answer.type,
    sdp: answer.sdp
  });

  // 10. Flush queued candidates
  for (const candidate of queuedCandidates) {
    await sendIceCandidate(sessionId, candidate);
  }
  queuedCandidates.length = 0;

  return { pc, dataChannel, sessionId };
}

async function sendIceCandidate(sessionId, candidate) {
  await fetch(`http://localhost:8000/api/v1/webrtc/offer/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      candidates: [{
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid,
        sdpMLineIndex: candidate.sdpMLineIndex
      }]
    })
  });
}
```

## Step-by-Step Breakdown

### 1. Get ICE Servers

```javascript
const iceResponse = await fetch("http://localhost:8000/api/v1/webrtc/ice-servers");
const { iceServers } = await iceResponse.json();
```

The server returns STUN/TURN server configuration. If TURN credentials are configured (via `HF_TOKEN` or Twilio), this enables connections through firewalls.

### 2. Create Peer Connection

```javascript
const pc = new RTCPeerConnection({ iceServers });
```

### 3. Create Data Channel

```javascript
const dataChannel = pc.createDataChannel("parameters", { ordered: true });
```

The data channel allows bidirectional communication:

- **Client → Server**: Send parameter updates (prompts, settings)
- **Server → Client**: Receive notifications (stream stopped, errors)

### 4. Add Input Video Track

```javascript
inputStream.getTracks().forEach((track) => {
  if (track.kind === "video") {
    pc.addTrack(track, inputStream);
  }
});
```

Unlike receive-only mode which uses `addTransceiver("video")`, video-to-video mode adds an actual video track from your input source. This can be:

- Webcam: `navigator.mediaDevices.getUserMedia({ video: true })`
- Screen capture: `navigator.mediaDevices.getDisplayMedia({ video: true })`
- File/canvas: Using a `<canvas>` element with `captureStream()`

### 5. Handle Incoming Track

```javascript
pc.ontrack = (event) => {
  if (event.streams[0]) {
    document.getElementById("outputVideo").srcObject = event.streams[0];
  }
};
```

### 6. Send Offer with Initial Parameters

```javascript
const response = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    initialParameters: {
      input_mode: "video",
      prompts: [{ text: "Your prompt here", weight: 1.0 }],
      denoising_step_list: [700, 500]
    }
  })
});
```

Note the `input_mode: "video"` parameter which tells the server to expect input video.

### 7. Complete Signaling

```javascript
const answer = await response.json();
await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
```

## Update Parameters During Streaming

After connection is established:

```javascript
function updatePrompt(newPrompt) {
  if (dataChannel.readyState === "open") {
    dataChannel.send(JSON.stringify({
      prompts: [{ text: newPrompt, weight: 1.0 }]
    }));
  }
}

// Smooth transition to new prompt
function transitionToPrompt(newPrompt, steps = 8) {
  dataChannel.send(JSON.stringify({
    transition: {
      target_prompts: [{ text: newPrompt, weight: 1.0 }],
      num_steps: steps
    }
  }));
}
```

## Stopping the Stream

```javascript
function stopStream(pc, dataChannel) {
  if (dataChannel) {
    dataChannel.close();
  }
  if (pc) {
    pc.close();
  }
}
```

## Error Handling

```javascript
dataChannel.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "stream_stopped") {
    if (data.error_message) {
      showError(data.error_message);
    }
    // Optionally attempt reconnection
    setTimeout(() => {
      startBidirectionalStream(inputStream, lastPrompt);
    }, 2000);
  }
};

pc.onconnectionstatechange = () => {
  if (pc.connectionState === "failed") {
    console.error("WebRTC connection failed");
    // Handle reconnection
  }
};
```

## VACE vs Standard V2V

When VACE is enabled on the pipeline (default), input video is routed through VACE for structural guidance. When disabled, input video is encoded and used for denoising.

```javascript
// Load with VACE disabled for traditional V2V
await loadPipeline("longlive", {
  vace_enabled: false
});
```

See [Using VACE](vace.md) for more details.

## Performance Tips

### Match Resolution

Set input resolution to match pipeline resolution for best quality:

```javascript
// If pipeline loaded with 512x512
const stream = await navigator.mediaDevices.getUserMedia({
  video: { width: 512, height: 512 }
});
```

### Frame Rate

Lower frame rates reduce bandwidth and processing load:

```javascript
video: { frameRate: { ideal: 15, max: 20 } }
```

## See Also

- [Receive Video](receive.md) - Text-to-video mode (no input)
- [Send Parameters](parameters.md) - Update parameters during streaming
- [Using VACE](vace.md) - Reference image conditioning
