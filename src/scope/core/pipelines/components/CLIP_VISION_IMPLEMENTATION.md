# CLIP Vision Implementation for Scope

## Overview

This document describes the CLIP Vision integration for the Daydream Scope project, enabling image conditioning for Wan video generation models.

## Current Implementation

### CLIP Vision Encoder (`clip_vision.py`)

Location: `src/scope/core/pipelines/clip_vision.py`

A standalone CLIP Vision encoder component that:

- Loads CLIP Vision models (ViT-huge-14 with 48 layers, 1664 hidden dim)
- Preprocesses images with CLIP normalization
- Encodes images to penultimate hidden states: `[B, 257, 1280]`
- Supports loading from OpenCLIP format checkpoints
- Compatible with the checkpoint at: `models/CLIP_Vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`

**Key Features:**
- Full PyTorch implementation (no ComfyUI dependencies)
- Efficient preprocessing with bicubic interpolation
- Returns penultimate hidden states needed for Wan models
- Supports both image files and tensors as input
- Automatic state dict conversion from OpenCLIP format

**Architecture:**
```python
CLIPVisionModel
├── CLIPVisionTransformer
│   ├── CLIPVisionEmbeddings (patch + position embeddings)
│   ├── pre_layrnorm
│   ├── encoder_layers (48 x CLIPEncoderLayer)
│   │   └── CLIPEncoderLayer
│   │       ├── layer_norm1
│   │       ├── CLIPVisionAttention
│   │       ├── layer_norm2
│   │       └── CLIPMLP
│   └── post_layernorm
└── visual_projection (1664 -> 1280)
```

### Test Scripts

1. **`test_clip_vision_spike.py`** - Validation spike test that:
   - Loads the CLIP Vision encoder
   - Encodes a test image
   - Validates output shapes and data
   - Documents integration requirements
   - Provides detailed next steps

2. **`test_clip_vision.py`** - Basic integration test (requires model_type='i2v')

## How CLIP Vision Works with Wan Models

### ComfyUI Reference

In ComfyUI, CLIP Vision is extensively used with Wan models:

1. **CLIP Vision Encoding:**
   ```python
   clip_vision_output = clip_vision.encode_image(image)
   # Returns: {'penultimate_hidden_states': [B, 257, 1280], ...}
   ```

2. **Conditioning:**
   ```python
   positive = conditioning_set_values(positive, {
       "clip_vision_output": clip_vision_output
   })
   ```

3. **Model Processing:**
   ```python
   # In CausalWanModel._forward_inference():
   if clip_fea is not None:
       context_clip = self.img_emb(clip_fea)  # [B, 257, 1280] -> [B, 257, dim]
       context = torch.concat([context_clip, context], dim=1)
   ```

### Model Requirements

For CLIP Vision to work, the `CausalWanModel` must be initialized with:

- `model_type="i2v"` - Enables the `img_emb` layer
- `img_emb = MLPProj(1280, dim)` - Projects CLIP features to model dimension

The `img_emb` layer is only initialized when `model_type == "i2v"`:
```python
if model_type == "i2v":
    self.img_emb = MLPProj(1280, dim)
```

## Integration Roadmap

### Phase 1: Core Integration (Completed)

- [x] CLIP Vision encoder component
- [x] Image preprocessing pipeline
- [x] State dict conversion for OpenCLIP format
- [x] Spike test for validation
- [x] Documentation

### Phase 2: Pipeline Integration (Next Steps)

#### 2.1 Model Configuration

Update `model.yaml` to support image-to-video mode:

```yaml
base_model_name: Wan2.1-T2V-1.3B
base_model_kwargs:
  model_type: i2v  # Add this to enable img_emb layer
  timestep_shift: 5.0
  sink_size: 3
```

Or pass during initialization:
```python
generator = WanDiffusionWrapper(
    CausalWanModel,
    model_name=base_model_name,
    model_dir=model_dir,
    generator_path=generator_path,
    generator_model_name=generator_model_name,
    model_type="i2v",  # Add this parameter
    **base_model_kwargs,
)
```

#### 2.2 Create CLIP Vision Conditioning Block

Create `wan2_1/blocks/clip_vision_conditioning.py`:

```python
class CLIPVisionConditioningBlock(ModularPipelineBlocks):
    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("clip_vision", CLIPVisionEncoder)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("image", required=False, type_hint=torch.Tensor),
            InputParam("clip_features", required=False, type_hint=torch.Tensor),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [OutputParam("clip_features", type_hint=torch.Tensor)]

    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)

        if block_state.clip_features is None and block_state.image is not None:
            # Encode image if not already encoded
            block_state.clip_features = components.clip_vision.encode_image(
                block_state.image
            )

        self.set_block_state(state, block_state)
        return components, state
```

#### 2.3 Modify DenoiseBlock

Update `wan2_1/blocks/denoise.py` to include CLIP features:

```python
# In DenoiseBlock.__call__():
conditional_dict = {
    "prompt_embeds": block_state.conditioning_embeds
}

# Add CLIP features if available
if block_state.clip_features is not None:
    conditional_dict["clip_fea"] = block_state.clip_features

# Pass to generator
_, denoised_pred = components.generator(
    noisy_image_or_video=noise,
    conditional_dict=conditional_dict,
    timestep=timestep,
    kv_cache=block_state.kv_cache,
    crossattn_cache=block_state.crossattn_cache,
    current_start=start_frame * frame_seq_length,
    current_end=end_frame * frame_seq_length,
    kv_cache_attention_bias=block_state.kv_cache_attention_bias,
)
```

#### 2.4 Modify WanDiffusionWrapper

Update `wan2_1/components/generator.py` to extract and pass CLIP features:

```python
def forward(
    self,
    noisy_image_or_video: torch.Tensor,
    conditional_dict: dict,
    timestep: torch.Tensor,
    # ... other params
) -> torch.Tensor:
    prompt_embeds = conditional_dict["prompt_embeds"]
    clip_fea = conditional_dict.get("clip_fea")  # Extract CLIP features

    # ... existing code ...

    flow_pred = self._call_model(
        noisy_image_or_video.permute(0, 2, 1, 3, 4),
        t=input_timestep,
        context=prompt_embeds,
        clip_fea=clip_fea,  # Pass CLIP features
        seq_len=self.seq_len,
        # ... other params
    ).permute(0, 2, 1, 3, 4)
```

#### 2.5 Update Pipeline Configuration

Add CLIP Vision component to pipeline:

```python
# In LongLivePipeline.__init__():
clip_vision_path = get_model_file_path(
    "CLIP_Vision/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)
clip_vision = CLIPVisionEncoder(
    checkpoint_path=clip_vision_path,
    device=device,
    dtype=dtype,
)
components.add("clip_vision", clip_vision)
```

#### 2.6 Update Block Sequence

Add CLIP Vision conditioning block to the pipeline:

```python
# In modular_blocks.py:
ALL_BLOCKS = InsertableDict([
    ("text_conditioning", TextConditioningBlock),
    ("clip_vision_conditioning", CLIPVisionConditioningBlock),  # Add this
    ("embedding_blending", EmbeddingBlendingBlock),
    # ... rest of blocks
])
```

### Phase 3: Testing and Validation

1. **Unit Tests:**
   - Test CLIP Vision encoder with various image sizes
   - Test state dict conversion
   - Test integration with CausalWanModel

2. **Integration Tests:**
   - Test image-to-video generation with CLIP conditioning
   - Compare outputs with/without CLIP features
   - Validate against ComfyUI outputs

3. **Performance Tests:**
   - Measure CLIP encoding overhead
   - Profile memory usage
   - Optimize preprocessing pipeline

### Phase 4: Production Ready

1. **Error Handling:**
   - Graceful fallback if img_emb layer not present
   - Validation of CLIP feature shapes
   - Clear error messages for configuration issues

2. **Documentation:**
   - Usage examples
   - API documentation
   - Migration guide from text-only to image+text conditioning

3. **Configuration:**
   - Add CLIP Vision to pipeline schema
   - Support optional vs required image conditioning
   - Allow disabling CLIP Vision at runtime

## Usage Examples

### Basic Usage (After Full Integration)

```python
from scope.core.pipelines.longlive.pipeline import LongLivePipeline
from PIL import Image

# Initialize pipeline with i2v model
config = OmegaConf.create({
    "model_dir": str(get_models_dir()),
    # ... other config
    "model_config": {
        "base_model_kwargs": {
            "model_type": "i2v"  # Enable CLIP Vision support
        }
    }
})

pipeline = LongLivePipeline(config, device="cuda", dtype=torch.bfloat16)

# Generate video from image + text prompt
image = Image.open("input.png")
prompts = [{"text": "A beautiful landscape with mountains", "weight": 100}]

output = pipeline(
    prompts=prompts,
    image=image  # Image conditioning via CLIP Vision
)
```

### Advanced Usage

```python
# Pre-encode CLIP features for reuse
from scope.core.pipelines.clip_vision import CLIPVisionEncoder

clip_vision = CLIPVisionEncoder(checkpoint_path, device="cuda")
clip_features = clip_vision.encode_image(image)

# Use pre-encoded features
output = pipeline(
    prompts=prompts,
    clip_features=clip_features  # Skip encoding step
)
```

## Technical Details

### CLIP Vision Model Specifications

- **Architecture:** ViT-huge-14
- **Parameters:** ~632M
- **Image Size:** 224x224
- **Patch Size:** 14x14
- **Layers:** 48
- **Hidden Dimension:** 1664
- **Num Heads:** 16
- **Intermediate Size:** 8192
- **Projection Dimension:** 1280
- **Output:** Penultimate hidden states [B, 257, 1280]
  - 1 class token + 256 patch tokens (16x16 patches)

### Data Flow

```
Input Image [H, W, 3]
    ↓
clip_preprocess() → [1, 3, 224, 224]
    ↓
CLIPVisionEmbeddings → [1, 257, 1664]
    ↓
48 x CLIPEncoderLayer → [1, 257, 1664]
    ↓
penultimate_hidden_states → [1, 257, 1664]
    ↓
visual_projection → [1, 257, 1280]
    ↓
img_emb (in model) → [1, 257, dim]
    ↓
concat with text_context → [1, 257 + text_len, dim]
    ↓
Model denoising with image + text conditioning
```

### Memory Considerations

- **CLIP Vision Model:** ~2.5 GB (bfloat16)
- **Single Image Encoding:** ~100 MB peak memory
- **Encoding Time:** ~50-100ms on GPU
- **Total Pipeline Memory:** +2.5 GB when CLIP Vision enabled

## References

### Implementation References

- **ComfyUI CLIP Vision:** `notes/ComfyUI/comfy/clip_vision.py`
- **ComfyUI Wan Nodes:** `notes/ComfyUI/comfy_extras/nodes_wan.py`
- **Scope CLIP Vision:** `src/scope/core/pipelines/clip_vision.py`

### Model Architecture References

- **CausalWanModel:** `src/scope/core/pipelines/longlive/modules/causal_model.py`
- **WanDiffusionWrapper:** `src/scope/core/pipelines/wan2_1/components/generator.py`
- **DenoiseBlock:** `src/scope/core/pipelines/wan2_1/blocks/denoise.py`

### Configuration References

- **LongLive Config:** `src/scope/core/pipelines/longlive/model.yaml`
- **Pipeline Setup:** `src/scope/core/pipelines/longlive/pipeline.py`

## Testing

Run the spike test to validate the CLIP Vision encoder:

```bash
cd src/scope/core/pipelines/longlive
python test_clip_vision_spike.py
```

Expected output:
- CLIP Vision model loads successfully
- Image encodes to shape [1, 257, 1280]
- Detailed integration instructions printed

## Future Enhancements

1. **Multi-Image Conditioning:**
   - Support for multiple reference images
   - Batch processing of images
   - Image interpolation for smooth transitions

2. **ControlNet Integration:**
   - Combine CLIP Vision with ControlNet
   - Support depth, canny, pose, etc.
   - Multi-modal conditioning

3. **Optimization:**
   - CLIP Vision quantization (fp8)
   - Caching of CLIP features
   - Lazy loading of CLIP Vision model

4. **Additional Models:**
   - Support for other CLIP variants (ViT-L, ViT-B)
   - SigLIP integration
   - Custom image encoders

## Status

**Current Status:** Phase 1 Complete (CLIP Vision Encoder)

**Ready For:** Spike testing and validation

**Next Milestone:** Phase 2 - Pipeline integration with i2v model support

**Estimated Effort:** 2-3 days for full integration and testing
