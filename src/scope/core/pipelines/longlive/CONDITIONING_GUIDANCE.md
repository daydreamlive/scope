# LongLive VACE Conditioning

This document describes the VACE conditioning feature for LongLive, a causal/autoregressive video model with VACE (Video-Aligned Conditioning Engine) support.

## Overview

LongLive supports flexible VACE conditioning with an implicit, elegant API. Simply provide what you want to use:

1. **Reference Images Only**: Static images (1-3 frames) for style/character consistency
2. **Conditioning Input Only**: Per-chunk guidance (depth, flow, pose, scribble, etc.) for structural control
3. **Both Combined**: Reference images + conditioning for style + structural guidance

The mode is implicit based on what you provide - no explicit mode parameter needed.

## Original VACE Architecture

Following the original VACE implementation (`notes/VACE/vace/models/wan/wan_vace.py`):
- Conditioning maps are treated as `input_frames` (3-channel RGB from annotators)
- Standard encoding path: `vace_encode_frames` (with masks=ones) -> `vace_encode_masks` -> `vace_latent`
- For conditioning: `masks = ones` (all white masks, goes through standard masking path)
- Reference images can be optionally combined with conditioning

## Architecture

### Key Components

#### 1. VaceEncodingBlock (`blocks/vace_encoding.py`)
- Handles encoding of VACE conditioning inputs
- Implicit mode detection based on what's provided
- Supports any combination of reference images and conditioning input
- No caching - application layer manages reuse

**Reference Images Only:**
- Encodes 1-3 static reference images
- No per-chunk conditioning
- Uses `vace_in_dim=96` (16 base × 6 for masking)

**Conditioning Input (depth, flow, pose, scribble, etc.):**
- Encodes 12 frames per chunk (matching output chunk size) via standard VACE path
- Conditioning maps are 3-channel RGB (from annotators)
- Standard encoding: `vace_encode_frames(vace_input, ref_images, masks=ones)` -> `vace_encode_masks(ones, ref_images)` -> `vace_latent`
- masks=ones goes through masking path: inactive (zeros) + reactive (conditioning) = 32 channels
- Generates fresh vace_context every chunk
- Uses `vace_in_dim=96` (32 channels from masking + 64 mask encoding = 96 total)

**Combined Mode:**
- Reference images provide style/character consistency
- Conditioning input provides structural guidance
- Both encoded together in single VACE context
- Regenerates hints every chunk (since conditioning changes)

#### 2. CausalVaceWanModel (`modules/causal_vace_model.py`)
- Extended with VACE blocks for hint injection
- `vace_regenerate_hints` parameter controls hint generation per chunk
- Application layer decides when to regenerate based on use case

### No Caching Philosophy

Unlike previous implementations, this design has **no internal caching**:
- Reference images are NOT cached in pipeline state
- Application layer sends inputs on every chunk where needed
- Benefits:
  - Simpler pipeline logic
  - Application has full control over what changes per chunk
  - Easy to switch conditioning mid-stream
  - No cache invalidation complexity

## Usage

### Configuration

```python
from omegaconf import OmegaConf
from scope.core.pipelines.longlive.pipeline import LongLivePipeline

# Configure pipeline with VACE support
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
            "vace_in_dim": 96,  # Standard VACE dimension
        },
    }),
    "height": 480,
    "width": 832,
})

device = torch.device("cuda")
pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
```

### Mode 1: Reference Images Only

```python
# Simple: just provide reference images
output = pipeline(
    prompts=[{"text": "Your prompt", "weight": 100}],
    ref_images=["path/to/ref1.png", "path/to/ref2.png"],
    vace_context_scale=1.0,
)
```

For multi-chunk generation with reference images, the application layer should send them on every chunk:

```python
outputs = []
for chunk in range(num_chunks):
    output = pipeline(
        prompts=[{"text": prompt, "weight": 100}],
        ref_images=ref_images,  # Send every chunk
        vace_context_scale=1.0,
    )
    outputs.append(output)
```

### Mode 2: Conditioning Input Only (Depth, Flow, Pose, Scribble)

```python
from scope.core.pipelines.longlive.vace_utils import preprocess_depth_frames, extract_depth_chunk

# Preprocess conditioning maps [F, H, W] -> [1, 3, F, H, W]
conditioning_video = preprocess_depth_frames(
    conditioning_maps,  # Can be depth, flow, pose, scribble, etc.
    config.height,
    config.width,
    device,
)

# Generate video chunk by chunk
outputs = []
for chunk_index in range(num_chunks):
    # Extract chunk for this generation step
    conditioning_chunk = extract_depth_chunk(
        conditioning_video,
        chunk_index,
        frames_per_chunk=12,
    )

    # Generate with conditioning
    output = pipeline(
        prompts=[{"text": "Your prompt", "weight": 100}],
        vace_input=conditioning_chunk,
        vace_context_scale=1.0,
    )
    outputs.append(output)
```

### Mode 3: Combined (Reference + Conditioning)

```python
# Combine reference images with conditioning for best of both worlds
for chunk_index in range(num_chunks):
    conditioning_chunk = extract_depth_chunk(
        conditioning_video,
        chunk_index,
        frames_per_chunk=12,
    )

    output = pipeline(
        prompts=[{"text": "Your prompt", "weight": 100}],
        ref_images=["path/to/ref1.png"],  # Style/character
        vace_input=conditioning_chunk,     # Structural guidance
        vace_context_scale=1.0,
    )
    outputs.append(output)
```

### Preprocessing Utilities

```python
from scope.core.pipelines.longlive.vace_utils import preprocess_depth_frames

# From numpy array [F, H, W]
conditioning_tensor = torch.from_numpy(conditioning_maps).float()

# Preprocess: resize, normalize, add batch/channel dims
conditioning_video = preprocess_depth_frames(
    conditioning_tensor,
    target_height=480,
    target_width=832,
    device=torch.device("cuda"),
)
# Output: [1, 3, F, H, W]
```

## Conditioning Types

The same API works for any type of conditioning that can be represented as image-like data:

- **Depth**: Depth maps from DepthAnything, MiDaS, etc.
- **Flow**: Optical flow from RAFT, FlowNet, etc.
- **Pose**: Human pose keypoints rendered as heatmaps
- **Scribble**: Hand-drawn scribbles or edge maps
- **Segmentation**: Semantic segmentation masks
- **Any other**: Any spatial guidance that can be represented as RGB images

All conditioning types use the same standard VACE encoding path with `masks=ones`.

## Technical Details

### Shape Validation

The VaceEncodingBlock performs extensive shape validation:

1. **Conditioning frame count**: Must be 12 frames per chunk
2. **Resolution**: Must match target height/width
3. **Latent frame count**: Must be 3 after VAE encoding
4. **Channel count**: Automatically handled (96 channels after full encoding)

### Error Messages

```
VaceEncodingBlock._encode_with_conditioning: Expected 12 frames
(num_frame_per_block=3 * vae_temporal_downsample_factor=4), got N frames
```
**Solution:** Ensure vace_input has exactly 12 frames per chunk.

```
VaceEncodingBlock._encode_with_conditioning: Input resolution HxW does not match
target resolution H'xW'
```
**Solution:** Resize conditioning maps to match pipeline resolution.

### Overlap Handling

LongLive automatically handles overlap between chunks:
- Chunk N: frames 0-11
- Chunk N+1: frames 9-20 (overlap at 9-11)

**For conditioning guidance:**
- Provide full 12 frames including overlap for each chunk
- LongLive's blending mechanism ensures smooth transitions

## Troubleshooting

### Issue: No conditioning effect

**Symptoms:** Generated video doesn't follow conditioning structure

**Diagnosis:**
- Check `vace_context_scale` (try 0.5-2.0 range)
- Verify conditioning maps are high quality
- Ensure conditioning maps are preprocessed correctly

**Solution:**
```python
# Adjust guidance strength
output = pipeline(
    ...,
    vace_context_scale=1.5,  # Increase for stronger guidance
)
```

### Issue: Reference images appearing in output

**Symptoms:** Reference images show up as first frames of video

**Diagnosis:**
- Check `use_dummy_frames` parameter
- Verify VaceEncodingBlock is using correct encoding mode

**Solution:**
```python
# Set use_dummy_frames=False to encode only reference images
output = pipeline(
    ...,
    ref_images=ref_images,
    use_dummy_frames=False,  # Default
)
```

## Implementation Notes

### Why No Caching?

**Previous Design:**
- Reference images cached in pipeline state
- Complex cache invalidation logic
- Application can't easily change inputs mid-stream

**Current Design:**
- Application layer sends inputs each chunk
- Pipeline is stateless regarding inputs
- Benefits:
  - Simpler pipeline code
  - Application has full control
  - Easy to change conditioning per chunk
  - No cache bugs

### Application Layer Optimization

If you want to reuse reference images or avoid reloading files:

```python
# Application-level caching (recommended)
class MyApplication:
    def __init__(self):
        self.ref_images_cache = ["path1.png", "path2.png"]

    def generate_chunk(self, chunk_index):
        # Reuse cached paths - pipeline will reload/encode each time
        # This is fine since encoding is cheap compared to generation
        return pipeline(
            ref_images=self.ref_images_cache,
            ...
        )
```

### Channel Padding Strategy

**Reference Only (vace_in_dim=96):**
```
ref_latents:    16 channels (VAE latent)
padding:        80 channels (zeros)
─────────────────────────────────────────
Total:          96 channels
```

**Conditioning (vace_in_dim=96):**
```
inactive:       16 channels (masked region - zeros)
reactive:       16 channels (masked region - conditioning)
mask_encoding:  64 channels (encoded masks)
─────────────────────────────────────────
Total:          96 channels
```

**Combined (vace_in_dim=96):**
```
ref_latents:    16 channels (VAE latent from refs)
inactive:       16 channels (masked region - zeros)
reactive:       16 channels (masked region - conditioning)
mask_encoding:  64 channels (encoded masks)
─────────────────────────────────────────
Total:         112 channels -> padded to match (ref prepended)
```

## Future Enhancements

### Potential Improvements

1. **Multi-conditioning**: Combine multiple conditioning types (depth + pose)
2. **Temporal smoothing**: Blend conditioning across chunks for smoother transitions
3. **Adaptive guidance**: Dynamically adjust `vace_context_scale` per chunk
4. **Conditioning refinement**: Use generated frames to improve conditioning iteratively

### Research Directions

1. **Temporal consistency**: Enforce temporal consistency across chunks
2. **Sparse conditioning**: Support sparse inputs with interpolation
3. **Controllable regions**: User-defined regions for selective guidance

## References

- **LongLive**: Causal/autoregressive video generation with KV caching
- **VACE**: Video-Aligned Conditioning Engine for video generation
- **DepthAnything**: Depth estimation for conditioning
- **ControlNet**: Similar conditioning approach for images
