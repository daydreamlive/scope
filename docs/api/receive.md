# Receive Video

This guide shows how to set up a WebRTC connection to receive video from the Scope API in **text-to-video mode** (no input video, just prompts).

## Overview

In receive-only mode:

- You send text prompts to control generation
- The server generates video and streams it back
- No input video required

## Prerequisites

1. Server is running: `uv run daydream-scope`
2. Models are downloaded for your pipeline
3. Pipeline is loaded (see [Load Pipeline](load.md))

## Complete Example

```javascript
async function startReceiveStream(initialPrompt = "A beautiful landscape") {
  const API_BASE = "http://localhost:8000";

  // 1. Get ICE servers from backend
  const iceResponse = await fetch(`${API_BASE}/api/v1/webrtc/ice-servers`);
  const { iceServers } = await iceResponse.json();

  // 2. Create peer connection
  const pc = new RTCPeerConnection({ iceServers });

  // State management
  let sessionId = null;
  const queuedCandidates = [];

  // 3. Create data channel for parameters
  const dataChannel = pc.createDataChannel("parameters", { ordered: true });

  dataChannel.onopen = () => {
    console.log("Data channel opened - ready for parameter updates");
  };

  dataChannel.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "stream_stopped") {
      console.log("Stream stopped:", data.error_message);
      pc.close();
    }
  };

  // 4. Add video transceiver (receive-only, no input)
  pc.addTransceiver("video");

  // 5. Handle incoming video track
  pc.ontrack = (event) => {
    if (event.streams && event.streams[0]) {
      const videoElement = document.getElementById("video");
      videoElement.srcObject = event.streams[0];
    }
  };

  // 6. Connection state monitoring
  pc.onconnectionstatechange = () => {
    console.log("Connection state:", pc.connectionState);
  };

  pc.oniceconnectionstatechange = () => {
    console.log("ICE state:", pc.iceConnectionState);
  };

  // 7. Handle ICE candidates (Trickle ICE)
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
        prompts: [{ text: initialPrompt, weight: 1.0 }],
        denoising_step_list: [1000, 750, 500, 250],
        manage_cache: true
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

  // 10. Send queued ICE candidates
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

### 4. Add Video Transceiver

```javascript
pc.addTransceiver("video");
```

For receive-only mode (no input video), we add a video transceiver instead of a track. This tells WebRTC we want to receive video.

### 5. Handle Incoming Track

```javascript
pc.ontrack = (event) => {
  if (event.streams[0]) {
    document.getElementById("video").srcObject = event.streams[0];
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
      prompts: [{ text: "Your prompt here", weight: 1.0 }],
      denoising_step_list: [1000, 750, 500, 250],
      manage_cache: true
    }
  })
});
```

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
      startReceiveStream(lastPrompt);
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

## See Also

- [Send and Receive Video](sendreceive.md) - Bidirectional video streaming
- [Send Parameters](parameters.md) - All available parameters
- [Load Pipeline](load.md) - Configure pipeline before streaming
