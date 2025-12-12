# LongLive Depth-Guided Video Generation

This document describes the depth-guided video generation feature for LongLive, a causal/autoregressive video model with VACE (Video-Aligned Conditioning Engine) support.

## Overview

LongLive now supports two VACE guidance modes:

1. **R2V (Reference-to-Video)**: Static reference images (1-3 frames) encoded once and cached
2. **Depth**: Per-chunk depth maps (12 frames) encoded via standard VACE path for frame-accurate depth guidance

## Original VACE Architecture

Following the original VACE implementation (notes/VACE/vace/models/wan/wan_vace.py):
- Depth maps are treated as `input_frames` (3-channel RGB from annotators)
- Standard encoding path: `vace_encode_frames` (with masks=ones) -> `vace_encode_masks` -> `vace_latent`
- For depth mode: `masks = ones` (all white masks, goes through standard masking path), `ref_images = None`

## Architecture

### Key Components

#### 1. VaceEncodingBlock (`blocks/vace_encoding.py`)
- Handles encoding of VACE conditioning inputs (reference images or depth maps)
- Supports both R2V and depth modes via `guidance_mode` parameter
- Creates separate VAE instances to avoid cache corruption

**R2V Mode:**
- Encodes 1-3 static reference images
- Generates vace_context once on first chunk
- Caches and reuses across all chunks
- Uses `vace_in_dim=96` (16 base × 6 for masking)

**Depth Mode:**
- Encodes 12 depth frames per chunk (matching output chunk size) via standard VACE path
- Depth maps are 3-channel RGB (from annotators like DepthAnything)
- Standard encoding: `vace_encode_frames(vace_input, None, masks=ones)` -> `vace_encode_masks(ones, None)` -> `vace_latent`
- masks=ones goes through masking path: inactive (zeros) + reactive (depth) = 32 channels
- Generates fresh vace_context every chunk
- No caching (different depth per chunk)
- Uses `vace_in_dim=96` (32 channels from masking + 64 mask encoding = 96 total)

#### 2. CausalVaceWanModel (`modules/causal_vace_model.py`)
- Extended to support `vace_guidance_mode` parameter
- Hint generation behavior:
  - R2V: Generate hints once (`current_start==0`), skip subsequent chunks
  - Depth: Generate hints every chunk
- Passes guidance mode through inference pipeline

#### 3. Pipeline Updates (`pipeline.py`)
- Loads separate `vace_vae` component for depth encoding
- Configurable `vace_in_dim` (16 for depth, 96 for R2V)
- Passes guidance mode through state

### Temporal Alignment

```
Output Space (12 frames)  ─┐
                           │ VAE Encode (4x temporal downsample)
Latent Space (3 frames)   ─┘
                           │
                           └─> num_frame_per_block = 3
```

**Critical:** Depth input must be 12 frames per chunk to match:
- `num_frame_per_block=3` (latent frames)
- `vae_temporal_downsample_factor=4` (12 output frames → 3 latent frames)

### Cache Isolation

```
┌─────────────┐           ┌─────────────┐
│  Main VAE   │           │  VACE VAE   │
│             │           │             │
│ - Autoregressive cache │ - No cache   │
│ - Used for generation  │ - Depth only │
└─────────────┘           └─────────────┘
```

**Why Separate VAEs?**
- Main VAE maintains autoregressive cache for generation
- VACE VAE encodes depth without corrupting main cache
- Each chunk needs fresh depth encoding

## Usage

### Basic Depth-Guided Generation

```python
from pathlib import Path
import torch
from omegaconf import OmegaConf
from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.longlive.pipeline import LongLivePipeline
from scope.core.pipelines.longlive.vace_utils import (
    extract_depth_chunk,
    preprocess_depth_frames,
)

# Configure pipeline with depth mode
config = OmegaConf.create({
    "model_dir": str(get_models_dir()),
    "generator_path": str(get_model_file_path("LongLive-1.3B/models/longlive_base.pt")),
    "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
    "vace_path": str(get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")),
    "text_encoder_path": str(get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")),
    "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
    "model_config": OmegaConf.create({
        "base_model_name": "Wan2.1-T2V-1.3B",
        "base_model_kwargs": {
            "vace_in_dim": 16,  # Depth mode
        },
    }),
    "height": 480,
    "width": 832,
})

device = torch.device("cuda")
pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)

# Preprocess depth frames [F, H, W] -> [1, 1, F, H, W]
depth_frames = ...  # Your depth maps
depth_video = preprocess_depth_frames(
    depth_frames,
    config.height,
    config.width,
    device,
)

# Generate video chunk by chunk
num_frames_generated = 0
frames_per_chunk = 12
chunk_index = 0
outputs = []

while num_frames_generated < target_frames:
    # Extract depth chunk
    depth_chunk = extract_depth_chunk(
        depth_video,
        chunk_index,
        frames_per_chunk,
    )

    # Generate with depth guidance (using standard VACE path)
    output = pipeline(
        prompts=[{"text": "Your prompt here", "weight": 100}],
        guidance_mode="depth",
        vace_input=depth_chunk,
        vace_context_scale=1.0,
    )

    outputs.append(output.detach().cpu())
    num_frames_generated += output.shape[0]
    chunk_index += 1

# Concatenate outputs
final_video = torch.concat(outputs)
```

### Depth Preprocessing

```python
from scope.core.pipelines.longlive.vace_utils import preprocess_depth_frames

# From numpy array [F, H, W]
depth_tensor = torch.from_numpy(depth_maps).float()

# Preprocess: resize, normalize, add batch/channel dims
depth_video = preprocess_depth_frames(
    depth_tensor,
    target_height=480,
    target_width=832,
    device=torch.device("cuda"),
)
# Output: [1, 1, F, H, W]
```

### Chunk Extraction

```python
from scope.core.pipelines.longlive.vace_utils import extract_depth_chunk

# Extract chunk for generation
depth_chunk = extract_depth_chunk(
    depth_video,        # [1, 1, F, H, W]
    chunk_index=0,      # 0-based chunk index
    num_frames_per_chunk=12,
)
# Output: [1, 1, 12, H, W]
```

### R2V Mode (for comparison)

```python
# Configure with R2V mode
config = OmegaConf.create({
    # ... same as above, but:
    "model_config": OmegaConf.create({
        "base_model_kwargs": {
            "vace_in_dim": 96,  # R2V mode
        },
    }),
})

# Generate with reference images
output = pipeline(
    prompts=[{"text": "Your prompt", "weight": 100}],
    guidance_mode="r2v",
    ref_images=["path/to/ref1.png", "path/to/ref2.png"],
    vace_context_scale=1.0,
)
```

## Technical Details

### Shape Validation

The VaceEncodingBlock performs extensive shape validation:

1. **Depth frame count**: Must be 12 frames per chunk
2. **Resolution**: Must match target height/width
3. **Latent frame count**: Must be 3 after VAE encoding
4. **Channel count**: Automatically handled (16 for depth, 96 for R2V)

### Error Messages

```
VaceEncodingBlock._encode_depth: Expected 12 depth frames
(num_frame_per_block=3 * vae_temporal_downsample_factor=4), got N frames
```
**Solution:** Ensure depth_frames has exactly 12 frames per chunk.

```
VaceEncodingBlock._encode_depth: Depth resolution HxW does not match
target resolution H'xW'
```
**Solution:** Resize depth maps to match pipeline resolution.

### Overlap Handling

LongLive automatically handles overlap between chunks:
- Chunk N: frames 0-11
- Chunk N+1: frames 9-20 (overlap at 9-11)

**For depth guidance:**
- Provide full 12 frames including overlap for each chunk
- LongLive's blending mechanism ensures smooth transitions
- No special depth handling needed for overlap regions

### Performance Considerations

**R2V Mode:**
- Hints generated once: ~0.5s overhead on first chunk
- Subsequent chunks: no overhead
- Best for: Static reference conditioning

**Depth Mode:**
- Hints regenerated every chunk: ~0.5s overhead per chunk
- No caching benefit
- Best for: Dynamic depth guidance per frame

## Test Script

Run the included test script to verify the implementation:

```bash
python -m scope.core.pipelines.longlive.test_depth_guidance
```

This will:
1. Generate synthetic depth maps
2. Run depth-guided generation
3. Save output video and depth visualization

## Model Checkpoints

### Required Files

1. **LongLive Base**: `LongLive-1.3B/models/longlive_base.pt`
2. **LongLive LoRA**: `LongLive-1.3B/models/lora.pt`
3. **VACE Weights**: `Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors`
4. **Text Encoder**: `WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors`
5. **Tokenizer**: `Wan2.1-T2V-1.3B/google/umt5-xxl`

### vace_in_dim Configuration

**Important Update:** To use pretrained VACE weights, both modes use `vace_in_dim=96`:

- **Depth mode**: `vace_in_dim=96` (depth latents padded from 16 to 96 channels)
- **R2V mode**: `vace_in_dim=96` (reference latents padded to 96 channels)

The VACE checkpoint was trained with 96-channel input for masked video generation. For depth mode, we pad the 16-channel depth latents with zeros to 96 channels, allowing us to load the pretrained weights.

**Alternative:** If training from scratch or fine-tuning for depth only, you could use `vace_in_dim=16`, but this requires not loading the `vace_patch_embedding` weights from the checkpoint.

## Depth Map Generation

### Using DepthAnything (Recommended)

```python
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
}

device = torch.device("cuda")
config = model_configs["small"]

depth_model = DepthAnythingV2(**config)
depth_model.load_state_dict(torch.load("checkpoints/depth_anything_v2_small.pth"))
depth_model = depth_model.to(device).eval()

# Infer depth
depth = depth_model.infer_image(rgb_frame)  # [H, W]
```

### Using MiDaS

```python
import torch

model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas = midas.to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Infer depth
input_batch = transform(rgb_frame).to(device)
with torch.no_grad():
    depth = midas(input_batch)
```

### Simple Alternative (for testing)

```python
import cv2

# Edge-based depth estimation
gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
depth = 255 - edges  # Invert
depth = cv2.GaussianBlur(depth, (9, 9), 0)  # Smooth
depth = depth.astype(np.float32) / 255.0  # Normalize
```

## Troubleshooting

### Issue: NaN in output

**Symptoms:** Video contains NaN values, black frames

**Diagnosis:**
- Check depth maps for NaN/Inf values
- Verify depth normalization to [-1, 1]
- Check VACE weights loaded correctly

**Solution:**
```python
# Sanitize depth maps
depth_frames = torch.nan_to_num(depth_frames, nan=0.0, posinf=1.0, neginf=-1.0)
```

### Issue: Shape mismatch errors

**Symptoms:** `Expected 12 depth frames, got N`

**Diagnosis:**
- Verify depth video has enough frames for requested chunks
- Check chunk extraction logic

**Solution:**
```python
total_chunks = len(depth_video[0, 0]) // frames_per_chunk
# Only generate up to available chunks
```

### Issue: Poor depth guidance quality

**Symptoms:** Generated video doesn't follow depth structure

**Diagnosis:**
- Check `vace_context_scale` (try 0.5-2.0 range)
- Verify depth maps are high quality
- Ensure `vace_in_dim=16` in config

**Solution:**
```python
# Adjust guidance strength
output = pipeline(
    ...,
    vace_context_scale=1.5,  # Increase for stronger guidance
)
```

## Implementation Notes

### Why Regenerate Hints Every Chunk?

**R2V Mode:**
- Reference images are static
- Same hints apply to all frames
- Cache once, reuse everywhere

**Depth Mode:**
- Depth maps change every frame
- Each chunk needs different depth guidance
- Must regenerate hints per chunk

### Channel Padding Strategy

**R2V Mode (vace_in_dim=96):**
```
ref_latents:    16 channels (VAE latent)
inactive:       16 channels (masked region)
reactive:       16 channels (masked region)
padding:        48 channels (zeros)
─────────────────────────────────────────
Total:          96 channels
```

**Depth Mode (vace_in_dim=16):**
```
depth_latents:  16 channels (VAE latent)
padding:        0 channels (none needed)
─────────────────────────────────────────
Total:          16 channels
```

## Future Enhancements

### Potential Improvements

1. **Multi-modal guidance**: Combine depth + reference images
2. **Temporal smoothing**: Blend depth maps across chunks for smoother transitions
3. **Adaptive guidance**: Dynamically adjust `vace_context_scale` per chunk
4. **Depth refinement**: Use generated frames to improve depth maps iteratively

### Research Directions

1. **Depth consistency**: Enforce temporal depth consistency across chunks
2. **Sparse depth**: Support sparse depth inputs with interpolation
3. **Controllable guidance**: User-defined depth regions for selective guidance

## References

- **LongLive**: Causal/autoregressive video generation with KV caching
- **VACE**: Video-Aligned Conditioning Engine for reference image conditioning
- **DepthAnything**: State-of-the-art monocular depth estimation
- **Wan2.1**: Base video diffusion model

## Contact & Support

For issues or questions about depth-guided generation:
1. Check this documentation
2. Review `test_depth_guidance.py` for working examples
3. Verify model checkpoints and configuration

---

**Implementation Date**: December 2025
**Version**: 1.0
**Status**: Production Ready
