# Send Parameters

After establishing a WebRTC connection, you can send real-time parameter updates via the data channel. This allows dynamic control of generation without reconnecting.

## Overview

Parameters are sent as JSON messages through the WebRTC data channel. Updates take effect on the next chunk that is generated.

## Setting Up the Data Channel

The data channel is created when establishing the WebRTC connection:

```javascript
// Create data channel before creating offer
const dataChannel = pc.createDataChannel("parameters", { ordered: true });

dataChannel.onopen = () => {
  console.log("Data channel ready for parameter updates");
};

dataChannel.onmessage = (event) => {
  const data = JSON.parse(event.data);

  // Handle notifications from server
  if (data.type === "stream_stopped") {
    console.log("Stream stopped:", data.error_message);
    pc.close();
  }
};
```

## Sending Parameter Updates

```javascript
function sendParameters(params) {
  if (dataChannel.readyState === "open") {
    dataChannel.send(JSON.stringify(params));
  } else {
    console.warn("Data channel not ready");
  }
}

// Example: Update prompt
sendParameters({
  prompts: [{ text: "A cat playing piano", weight: 1.0 }]
});

// Example: Update multiple parameters
sendParameters({
  prompts: [{ text: "A sunset over the ocean", weight: 1.0 }],
  denoising_step_list: [800, 600, 400],
  noise_scale: 0.7
});
```

## Available Parameters

### Prompts

Control what is being generated:

```javascript
// Single prompt
sendParameters({
  prompts: [{ text: "A beautiful forest", weight: 1.0 }]
});

// Blended prompts (spatial blending within frame)
sendParameters({
  prompts: [
    { text: "A sunny day", weight: 0.7 },
    { text: "A rainy day", weight: 0.3 }
  ],
  prompt_interpolation_method: "linear"  // or "slerp"
});
```

| Parameter                     | Type   | Default    | Description                                 |
| ----------------------------- | ------ | ---------- | ------------------------------------------- |
| `prompts`                     | array  | -          | Array of `{ text: string, weight: number }` |
| `prompt_interpolation_method` | string | `"linear"` | `"linear"` or `"slerp"` for blending        |

### Prompt Transitions

Smoothly transition between prompts over multiple frames:

```javascript
sendParameters({
  transition: {
    target_prompts: [
      { text: "A night sky with stars", weight: 1.0 }
    ],
    num_steps: 8,  // Transition over 8 chunks
    temporal_interpolation_method: "linear"
  }
});
```

| Parameter                                  | Type   | Default    | Description                             |
| ------------------------------------------ | ------ | ---------- | --------------------------------------- |
| `transition.target_prompts`                | array  | -          | Target prompts to transition to         |
| `transition.num_steps`                     | int    | 4          | Frames to transition over (0 = instant) |
| `transition.temporal_interpolation_method` | string | `"linear"` | `"linear"` or `"slerp"`                 |

### Denoising Steps

Control quality vs speed tradeoff:

```javascript
// More steps = higher quality, slower
sendParameters({
  denoising_step_list: [1000, 750, 500, 250]
});

// Fewer steps = faster, lower quality
sendParameters({
  denoising_step_list: [700, 400]
});
```

| Parameter             | Type  | Description                                     |
| --------------------- | ----- | ----------------------------------------------- |
| `denoising_step_list` | array | Descending timesteps (e.g., `[1000, 750, 500]`) |

### Noise Control

```javascript
sendParameters({
  noise_scale: 0.8,      // 0.0-1.0, amount of noise
  noise_controller: true  // Auto-adjust based on motion
});
```

| Parameter          | Type  | Range   | Description                       |
| ------------------ | ----- | ------- | --------------------------------- |
| `noise_scale`      | float | 0.0-1.0 | Manual noise amount               |
| `noise_controller` | bool  | -       | Enable automatic noise adjustment |

### Cache Control

```javascript
// Enable automatic cache management
sendParameters({
  manage_cache: true
});

// Or manual cache reset
sendParameters({
  reset_cache: true  // Trigger one-time cache reset
});
```

| Parameter      | Type | Range | Description                  |
| -------------- | ---- | ----- | ---------------------------- |
| `manage_cache` | bool | -     | Auto cache management        |
| `reset_cache`  | bool | -     | Force cache reset (one-shot) |

### Playback Control

```javascript
// Pause generation
sendParameters({ paused: true });

// Resume generation
sendParameters({ paused: false });
```

### LoRA Scale Updates

Update LoRA adapter scales at runtime (requires `lora_merge_mode: "runtime_peft"` at load time):

```javascript
sendParameters({
  lora_scales: [
    { path: "/path/to/style.safetensors", scale: 0.5 },
    { path: "/path/to/character.safetensors", scale: 1.2 }
  ]
});
```

### VACE Parameters

Control strength of visual conditioning:

```javascript
sendParameters({
  vace_ref_images: ["/path/to/reference.png"],
  vace_context_scale: 1.0  // 0.0-2.0
});
```

See [Using VACE](vace.md) for detailed VACE usage.

### Spout (Windows)

Enable Spout output for sending frames to external applications:

```javascript
sendParameters({
  spout_sender: {
    enabled: true,
    name: "ScopeOutput"
  }
});

// Receive from Spout instead of WebRTC input
sendParameters({
  spout_receiver: {
    enabled: true,
    name: "ExternalApp"
  }
});
```

## Initial Parameters

You can also send initial parameters when establishing the WebRTC connection:

```javascript
const response = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    initialParameters: {
      prompts: [{ text: "Initial prompt", weight: 1.0 }],
      denoising_step_list: [1000, 750, 500, 250],
      manage_cache: true
    }
  })
});
```

## See Also

- [Receive Video](receive.md) - Set up WebRTC connection
- [Using VACE](vace.md) - Reference image conditioning
- [Load Pipeline](load.md) - Configure pipeline at load time
