# Scope Optical Flow TensorRT Plugin

TensorRT-accelerated optical flow preprocessor for Scope VACE conditioning.

Ported from StreamDiffusion's `TemporalNetTensorRTPreprocessor`. Uses RAFT (Recurrent All-Pairs Field Transforms) from torchvision, compiled to TensorRT with optimization profiles for real-time performance.

## Features

- **UI Integration**: Appears as "Flow" in the VACE Conditioning dropdown
- **Auto-compilation**: Automatically exports RAFT to ONNX and builds TensorRT engine on first use
- **GPU-specific caching**: TensorRT engines are cached per GPU model and resolution
- **VACE-compatible output**: RGB flow visualization in the format expected by VACE conditioning
- **Dynamic shape support**: Handles buffer reallocation for different input sizes

## Installation

```bash
cd plugins/scope-optical-flow-tensorrt
pip install -e .
```

## Usage

### With Scope

Once installed, the preprocessor automatically registers with Scope and appears as "Flow" in the VACE Conditioning dropdown in the UI.

### Standalone

```python
from scope_optical_flow_tensorrt import OpticalFlowTensorRTPreprocessor

# Initialize preprocessor (lazy loads on first use)
preprocessor = OpticalFlowTensorRTPreprocessor(
    height=512,           # Flow computation height
    width=512,            # Flow computation width
    device="cuda",
    fp16=True,
    flow_strength=1.0,    # Flow visualization intensity (0.0-2.0)
)

# Process video frames
# Input: [B, C, F, H, W] or [C, F, H, W] in [0, 1] range
flow_maps = preprocessor(input_frames)
# Output: [1, C, F, H, W] in [-1, 1] range

# Reset between different video sequences
preprocessor.reset()
```

## Input/Output Format

### Input
- Shape: `[B, C, F, H, W]` or `[C, F, H, W]` or `[F, H, W, C]`
- Range: `[0, 1]` float
- Channels: RGB (3 channels)

### Output
- Shape: `[1, C, F, H, W]`
- Range: `[-1, 1]` float
- Content: RGB flow visualization where:
  - Hue encodes flow direction
  - Saturation encodes flow magnitude
  - First frame outputs zero flow (no motion reference)

## How It Works

1. **ONNX Export**: On first use, exports RAFT Small from `torchvision.models.optical_flow` to ONNX
2. **TensorRT Build**: Compiles ONNX to TensorRT engine with optimization profiles (FP16 by default)
3. **Frame Pairs**: Computes optical flow between consecutive frames using TensorRT inference
4. **Flow Visualization**: Converts raw flow vectors to RGB using `torchvision.utils.flow_to_image`
5. **First Frame**: Since there's no previous frame, the first frame outputs zero flow

## Model Storage

Models are stored in `~/.daydream-scope/models/optical-flow-tensorrt/`:
- `raft_small_{H}x{W}.onnx` - Exported RAFT ONNX model
- `raft_small_{H}x{W}_{gpu_name}.trt` - GPU-specific TensorRT engine

Override with `DAYDREAM_SCOPE_MODELS_DIR` environment variable.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- TensorRT >= 10.0
- polygraphy >= 0.49

## License

Apache-2.0
