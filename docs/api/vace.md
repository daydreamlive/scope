# Using VACE

VACE (Video All-In-One Creation and Editing) enables guiding generation using reference images and control videos. This guide shows how to use VACE for style transfer, character consistency, and video transformation.

## Overview

VACE allows you to:

- Condition generation on reference images (style, character, scene)
- Use control videos to preserve structure and motion
- Extend video from first and/or last frames (FFLF extension mode)
- Inpaint specific regions using masks
- Control the influence strength of visual conditioning

## Prerequisites

1. Pipeline loaded with VACE enabled (default):

```javascript
await loadPipeline("longlive", {
  vace_enabled: true  // This is the default
});
```

2. For reference images: Images uploaded or available in the assets directory
3. For control videos: A video input source (webcam, screen capture, or file)

## Uploading Reference Images

Upload images via the assets API:

```javascript
async function uploadReferenceImage(file) {
  const arrayBuffer = await file.arrayBuffer();
  const filename = encodeURIComponent(file.name);

  const response = await fetch(
    `http://localhost:8000/api/v1/assets?filename=${filename}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: arrayBuffer
    }
  );

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return await response.json();
}

// Usage
const fileInput = document.getElementById("imageInput");
const file = fileInput.files[0];
const assetInfo = await uploadReferenceImage(file);
console.log("Uploaded:", assetInfo.path);
// Returns: { path: "/path/to/assets/image.png", ... }
```

## Setting Reference Images

### Via Initial Parameters

Set reference images when starting the WebRTC connection:

```javascript
const response = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    initialParameters: {
      prompts: [{ text: "A person walking in a forest", weight: 1.0 }],
      vace_ref_images: ["/path/to/reference.png"],
      vace_context_scale: 1.0
    }
  })
});
```

### Via Data Channel

Update reference images during streaming:

```javascript
// Set new reference image
dataChannel.send(JSON.stringify({
  vace_ref_images: ["/path/to/new_reference.png"],
  vace_context_scale: 1.0
}));

// Multiple reference images
dataChannel.send(JSON.stringify({
  vace_ref_images: [
    "/path/to/style_ref.png",
    "/path/to/character_ref.png"
  ],
  vace_context_scale: 1.2
}));

// Clear reference images
dataChannel.send(JSON.stringify({
  vace_ref_images: []
}));
```

## VACE Parameters

| Parameter            | Type   | Range   | Default | Description                                      |
| -------------------- | ------ | ------- | ------- | ------------------------------------------------ |
| `vace_ref_images`    | array  | -       | `[]`    | List of reference image paths                    |
| `vace_context_scale` | float  | 0.0-2.0 | 1.0     | Visual conditioning strength                     |
| `first_frame_image`  | string | -       | `null`  | Path to first frame for FFLF extension mode      |
| `last_frame_image`   | string | -       | `null`  | Path to last frame for FFLF extension mode       |
| `vace_input_masks`   | tensor | -       | `null`  | Masks for inpainting (1=generate, 0=preserve)    |

### Context Scale

The `vace_context_scale` controls how strongly reference images influence generation:

- **0.0**: No reference influence (pure text-to-video)
- **0.5**: Subtle influence, more creative freedom
- **1.0**: Balanced influence (default)
- **1.5**: Strong influence, closer to reference
- **2.0**: Maximum influence, may reduce diversity

```javascript
// Subtle style influence
dataChannel.send(JSON.stringify({
  vace_ref_images: ["/path/to/style.png"],
  vace_context_scale: 0.5
}));

// Strong character preservation
dataChannel.send(JSON.stringify({
  vace_ref_images: ["/path/to/character.png"],
  vace_context_scale: 1.5
}));
```

## Using Control Videos

When VACE is enabled, you can send a control video to guide generation while preserving the structure and motion of your input. This works the same way as sending video in regular video-to-video mode.

### Sending Control Video

Set up a WebRTC connection with video input, just as you would for [Send and Receive Video](sendreceive.md):

```javascript
// Get input video (webcam, screen capture, or file)
const inputStream = await navigator.mediaDevices.getUserMedia({ video: true });

// Add video track to peer connection
inputStream.getTracks().forEach((track) => {
  if (track.kind === "video") {
    pc.addTrack(track, inputStream);
  }
});

// Send offer with video input mode
const response = await fetch("http://localhost:8000/api/v1/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: offer.sdp,
    type: offer.type,
    initialParameters: {
      input_mode: "video",
      prompts: [{ text: "A cyberpunk city scene", weight: 1.0 }]
    }
  })
});
```

When VACE is enabled (the default), the input video is routed through VACE for structural guidance. The generated output will follow the motion and composition of your input video while applying the style and content from your prompts.

### Combining Control Video with Reference Images

You can use both control video and reference images together for maximum control:

```javascript
// Send control video via WebRTC track (as shown above)
// Then set reference images via data channel
dataChannel.send(JSON.stringify({
  vace_ref_images: ["/path/to/style_reference.png"],
  vace_context_scale: 1.0
}));
```

This allows you to:
- Use the control video for motion and structure
- Use reference images for style, character appearance, or scene elements

## First Frame Last Frame (FFLF) Extension Mode

FFLF extension mode generates video that connects reference frames at the start and/or end of the sequence. This is useful for creating smooth transitions or extending existing video clips.

You can provide `first_frame_image`, `last_frame_image`, or both:

- `first_frame_image`: Generate video extending after this frame
- `last_frame_image`: Generate video extending before this frame
- Both: Generate video connecting the two frames

### Usage Examples

```javascript
// Extend from first frame
dataChannel.send(JSON.stringify({
  first_frame_image: "/path/to/start_frame.png",
  prompts: [{ text: "A person walking through a forest", weight: 1.0 }]
}));

// Extend to last frame
dataChannel.send(JSON.stringify({
  last_frame_image: "/path/to/end_frame.png",
  prompts: [{ text: "A person walking through a forest", weight: 1.0 }]
}));

// Connect first and last frames
dataChannel.send(JSON.stringify({
  first_frame_image: "/path/to/start_frame.png",
  last_frame_image: "/path/to/end_frame.png",
  prompts: [{ text: "A smooth transition between scenes", weight: 1.0 }]
}));
```

## Listing Available Assets

Get existing assets in the assets directory:

```javascript
async function listAssets(type = "image") {
  const response = await fetch(`http://localhost:8000/api/v1/assets?type=${type}`);
  return await response.json();
}

const { assets } = await listAssets("image");
console.log("Available reference images:", assets);
// [{ name: "ref1", path: "/path/to/ref1.png", ... }, ...]
```

## See Also

- [Receive Video](receive.md) - Text-to-video mode
- [Send and Receive Video](sendreceive.md) - Video-to-video mode
- [Send Parameters](parameters.md) - All available parameters
