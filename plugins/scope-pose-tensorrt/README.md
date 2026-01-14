# Scope Pose TensorRT Plugin

TensorRT-accelerated pose estimation preprocessor for Scope's VACE/ControlNet conditioning system.

## Features

- **UI Integration**: Shows up as "Pose" in VACE Conditioning dropdown
- **Automatic model download**: Downloads YoloNas Pose ONNX model from HuggingFace on first use
- **Automatic TRT compilation**: Compiles ONNX to GPU-specific TensorRT engine
- **VACE-compatible output**: Produces pose maps in the correct format for VACE conditioning
- **Lazy initialization**: No startup cost until first inference

## Installation

```bash
# Install from local source (editable mode)
daydream-scope install -e plugins/scope-pose-tensorrt

# Or if published to PyPI
daydream-scope install scope-pose-tensorrt
```

Note: The `daydream-scope install` command validates dependencies before installing to prevent environment conflicts. Use `--force` to skip validation if needed.

### Requirements

- Python >= 3.10
- CUDA-capable GPU
- TensorRT >= 10.0
- polygraphy >= 0.49

## Usage

### Via UI (Recommended)

1. Install the plugin
2. Start Scope with a VACE-enabled pipeline (e.g., LongLive)
3. Enable VACE and set input mode to Video
4. In VACE settings, set "Conditioning" dropdown to "Pose"
5. Your camera/video input will be processed through pose detection

### Programmatic Usage

```python
from scope_pose_tensorrt import PoseTensorRTPreprocessor

# Initialize (lazy - no model loaded yet)
preprocessor = PoseTensorRTPreprocessor(
    detect_resolution=640,  # Match ONNX model input
    device="cuda",
    fp16=True,  # Use FP16 for TRT compilation
)

# Optional: warm up to avoid latency on first frame
preprocessor.warmup()

# Process video frames
# Input: [B, C, F, H, W] or [C, F, H, W] in [0, 1] range
pose_maps = preprocessor(input_frames)
# Output: [1, C, F, H, W] in [-1, 1] range

# Use with VACE-enabled Scope pipeline
pipeline(
    vace_input_frames=pose_maps,
    vace_input_masks=None,  # Defaults to all ones
    ...
)
```

## How It Works

1. **First run**: Downloads ONNX model from HuggingFace, compiles to TensorRT engine
2. **Subsequent runs**: Loads cached TRT engine (GPU-specific)
3. **Inference**: Runs pose detection, renders skeleton on black background
4. **Output**: Converts to VACE-compatible tensor format

## Model Storage

Models are stored in:
- `~/.daydream-scope/models/pose-tensorrt/` (default)
- Or set `DAYDREAM_SCOPE_MODELS_DIR` environment variable

TRT engines are cached per-GPU (e.g., `yolo_nas_pose_l_nvidia_geforce_rtx_4090.trt`).

## Output Format

- **Shape**: `[1, 3, F, H, W]` (batch=1, RGB, F frames)
- **Range**: `[-1.0, 1.0]` float32
- **Content**: Colored skeleton on black background (COCO format)

## License

Apache 2.0
