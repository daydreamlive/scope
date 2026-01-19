# Load Pipeline

Before streaming video, you need to load a pipeline. Pipeline loading is asynchronous - the API initiates loading and returns immediately, so you must poll for completion.

## Overview

Loading a pipeline:

1. Loads model weights to GPU memory
2. Initializes the inference pipeline

## Load a Pipeline

```javascript
async function loadPipeline(request) {
  const response = await fetch("http://localhost:8000/api/v1/pipeline/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to load pipeline: ${error}`);
  }

  return await response.json();
}

// Example: Load longlive with default settings
await loadPipeline({
  pipeline_ids: ["longlive"]
});

// Example: Load with custom resolution
await loadPipeline({
  pipeline_ids: ["longlive"],
  load_params: {
    height: 512,
    width: 512,
    seed: 42
  }
});

// Example: Load with preprocessor
await loadPipeline({
  pipeline_ids: ["video-depth-anything", "longlive"],
  load_params: {
    height: 512,
    width: 512,
    seed: 42
  }
});
```

### Request Body

```json
{
  "pipeline_ids": ["longlive"],
  "load_params": {
    "height": 320,
    "width": 576,
    "seed": 42,
    "quantization": null,
    "vace_enabled": false,
    "vae_type": "wan",
    "loras": [
      {
        "path": "/path/to/lora.safetensors",
        "scale": 1.0
      }
    ],
    "lora_merge_mode": "permanent_merge"
  }
}
```

To load a pipeline with a preprocessor, include the preprocessor ID before the main pipeline in the `pipeline_ids` array:

```json
{
  "pipeline_ids": ["video-depth-anything", "longlive"],
  "load_params": {
    "height": 320,
    "width": 576,
    "seed": 42,
    "vace_enabled": true
  }
}
```

### Load Parameters

| Parameter         | Type   | Default             | Description                             |
| ----------------- | ------ | ------------------- | --------------------------------------- |
| `height`          | int    | 320                 | Output height (16-2048)                 |
| `width`           | int    | 576                 | Output width (16-2048)                  |
| `seed`            | int    | 42                  | Random seed for generation              |
| `vace_enabled`    | bool   | true                | Enable VACE                             |
| `vae_type`        | string | `"wan"`             | VAE type (see [VAE Types](#vae-types))  |
| `loras`           | array  | null                | LoRA adapters to load                   |
| `lora_merge_mode` | string | `"permanent_merge"` | `"permanent_merge"` or `"runtime_peft"` |

## Check Pipeline Status

Poll the status endpoint to know when loading completes:

```javascript
async function getPipelineStatus() {
  const response = await fetch("http://localhost:8000/api/v1/pipeline/status");

  if (!response.ok) {
    throw new Error(`Failed to get status: ${response.statusText}`);
  }

  return await response.json();
}

const status = await getPipelineStatus();
console.log(status);
```

### Response Format

```json
{
  "status": "loaded",
  "pipeline_id": "longlive",
  "load_params": {
    "height": 320,
    "width": 576,
    "seed": 42
  },
  "loaded_lora_adapters": [
    { "path": "/path/to/lora.safetensors", "scale": 1.0 }
  ],
  "error": null
}
```

### Status Values

| Status       | Description                          |
| ------------ | ------------------------------------ |
| `not_loaded` | No pipeline loaded                   |
| `loading`    | Pipeline is loading (wait and retry) |
| `loaded`     | Pipeline ready for streaming         |
| `error`      | Loading failed (check `error` field) |

## Wait for Pipeline Ready

Complete example that loads a pipeline and waits for it to be ready:

```javascript
async function waitForPipelineLoaded(timeoutMs = 300000, pollIntervalMs = 1000) {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    const status = await getPipelineStatus();

    switch (status.status) {
      case "loaded":
        console.log("Pipeline loaded successfully");
        return status;

      case "loading":
        console.log("Pipeline loading...");
        await new Promise(r => setTimeout(r, pollIntervalMs));
        break;

      case "error":
        throw new Error(`Pipeline load failed: ${status.error}`);

      case "not_loaded":
        throw new Error("Pipeline not loading - was load request sent?");
    }
  }

  throw new Error("Timeout waiting for pipeline to load");
}

// Complete workflow
async function loadAndWait(pipelineIds, loadParams = {}) {
  console.log(`Loading ${pipelineIds.join(" -> ")}...`);
  await loadPipeline({
    pipeline_ids: pipelineIds,
    load_params: loadParams
  });
  return await waitForPipelineLoaded();
}

// Usage: Load single pipeline
const status = await loadAndWait(["longlive"], { height: 320, width: 576 });
console.log(`Pipeline ${status.pipeline_id} ready`);

// Usage: Load with preprocessor
const statusWithPreprocessor = await loadAndWait(
  ["video-depth-anything", "longlive"],
  { height: 320, width: 576, vace_enabled: true }
);
console.log(`Pipeline ${statusWithPreprocessor.pipeline_id} ready`);
```

## Switching Pipelines

When you load a different pipeline (or the same pipeline with different parameters), the previous pipeline is automatically unloaded:

```javascript
// Load longlive
await loadAndWait(["longlive"]);

// This automatically unloads longlive and loads streamdiffusionv2
await loadAndWait(["streamdiffusionv2"]);

// Load with preprocessor
await loadAndWait(["video-depth-anything", "longlive"], {
  vace_enabled: true
});
```

## LoRA Configuration

Load LoRA adapters at pipeline load time:

```javascript
await loadPipeline({
  pipeline_ids: ["longlive"],
  load_params: {
    loras: [
      {
        path: "/path/to/style.safetensors",
        scale: 0.8
      },
      {
        path: "/path/to/character.safetensors",
        scale: 1.0
      }
    ],
    // permanent_merge: Maximum FPS, no runtime updates
    // runtime_peft: Allows runtime scale updates, lower FPS
    lora_merge_mode: "runtime_peft"
  }
});
```

With `runtime_peft` mode, you can update LoRA scales during streaming via the data channel. See [Send Parameters](parameters.md) for details.

## Preprocessor Configuration

Preprocessors can be configured to automatically process input videos before they reach the main pipeline. For example, the `video-depth-anything` preprocessor can generate depth maps from source videos in real-time for use with VACE V2V.

To configure a preprocessor, include it in the `pipeline_ids` array before the main pipeline:

```javascript
// Load longlive with video-depth-anything preprocessor
await loadPipeline({
  pipeline_ids: ["video-depth-anything", "longlive"],
  load_params: {
    height: 320,
    width: 576,
    seed: 42
  }
});
```

The preprocessor will automatically process input videos and pass the results to the main pipeline. When using VACE V2V with a preprocessor, the preprocessor's output (e.g., depth maps) will be used as the control video.

```javascript
// Example: Load with preprocessor for real-time depth estimation
async function loadWithPreprocessor(pipelineId, preprocessorId, loadParams = {}) {
  const pipelineIds = [];

  if (preprocessorId) {
    pipelineIds.push(preprocessorId);
  }
  pipelineIds.push(pipelineId);

  return await loadPipeline({
    pipeline_ids: pipelineIds,
    load_params: loadParams
  });
}

// Usage: Load longlive with depth preprocessor
await loadWithPreprocessor("longlive", "video-depth-anything", {
  height: 320,
  width: 576,
  vace_enabled: true
});
```

> [!NOTE]
> Currently, only the `video-depth-anything` preprocessor is available. Additional preprocessors will be available via plugins in the future.

## VAE Types

The `vae_type` parameter controls which VAE (Variational Autoencoder) is used for encoding pixels to latents and decoding latents to pixels. Different VAE types offer tradeoffs between quality, speed, and memory usage.

| Type       | Quality | Speed  |
| ---------- | ------- | ------ |
| `wan`      | Best    | Slow   |
| `lightvae` | High    | Medium |
| `tae`      | Average | Fast   |
| `lighttae` | High    | Fast   |

See [VAE Types](../vae.md) for detailed descriptions of each type.

```javascript
await loadPipeline({
  pipeline_ids: ["longlive"],
  load_params: {
    vae_type: "lighttae"
  }
});
```

## Error Handling

```javascript
try {
  await loadPipeline({
    pipeline_ids: ["longlive"]
  });
  await waitForPipelineLoaded();
} catch (error) {
  if (error.message.includes("CUDA")) {
    console.error("GPU error - check CUDA availability");
  } else if (error.message.includes("not found")) {
    console.error("Models not downloaded - run download first");
  } else {
    console.error("Load failed:", error.message);
  }
}
```

## See Also

- [Receive Video](receive.md) - Start streaming after pipeline loads
- [Send Parameters](parameters.md) - Update parameters during streaming
