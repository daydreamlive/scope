# VACE Implementation Design

This document outlines the key architectural differences between the Scope VACE implementation and the original VACE implementation from Alibaba's repository.

## Overview

Our VACE (All-In-One Video Creation and Editing) implementation adapts the original design for causal/streaming generation in the Wan2.1 pipeline. While both implementations use the same hint injection mechanism at the transformer level, we diverge fundamentally in how reference images are integrated with video latents during the diffusion process.

## Key Architectural Difference: Latent Space Separation

### Original VACE: Reference Frames in Diffusion Latent Space

**Encoding (notes/VACE/vace/models/wan/wan_vace.py:153):**
```python
# Reference latents concatenated with video latents in temporal dimension
latent = torch.cat([*ref_latent, latent], dim=1)
```

If you have 2 reference images and 20 video frames, the concatenated latent has shape `[channels, 22 frames, H, W]`.

**Noise Generation (wan_vace.py:343-354):**
```python
target_shape = list(z0[0].shape)  # e.g., [32, 22, H, W] with refs
target_shape[0] = int(target_shape[0] / 2)  # [16, 22, H, W]
noise = torch.randn(target_shape[0], target_shape[1], ...)  # Noise for ALL 22 frames
```

The noise tensor includes positions for reference frames. The diffusion model denoises a unified latent space containing both reference and video frame positions.

**Decoding (wan_vace.py:252-256):**
```python
# Must strip reference frames before VAE decode
trimed_zs = []
for z, refs in zip(zs, ref_images):
    if refs is not None:
        z = z[:, len(refs):, :, :]  # Remove first N frames
    trimed_zs.append(z)
return vae.decode(trimed_zs)
```

**Architecture:** Reference frames are part of the diffusion latent space. The denoising process operates on a unified tensor containing both reference and video frames temporally concatenated.

### Our Implementation: Reference Frames in Conditioning Space Only

**Encoding (src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py:236-237):**
```python
# R2V mode: vace_context contains ONLY reference images
vace_context = [ref_latent_batch]  # Shape: [96, num_refs, H, W]
```

For conditioning mode (depth/flow/etc.), references ARE concatenated with conditioning frames in `vace_context` (via vace_utils.vace_encode_frames line 95), but this concatenated tensor is used ONLY for hint generation, not for diffusion latent space.

**Noise Generation (src/scope/core/pipelines/wan2_1/blocks/prepare_latents.py:104-115):**
```python
# Noise generated ONLY for video frames (no reference frame positions)
latents = torch.randn([
    1,  # batch_size
    num_latent_frames,  # e.g., 3 latent frames (= 12 output frames)
    16,
    latent_height,
    latent_width,
], ...)
```

The noise tensor contains NO reference frame positions. It's sized only for the video frames to be generated.

**Denoising Flow (src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py:342-354):**
```python
# vace_context processed separately to generate hints
hints = self.forward_vace(x, vace_context, seq_len, ...)

# Main denoising operates ONLY on video latents (x), not vace_context
kwargs = {
    "hints": hints,
    "context_scale": vace_context_scale,
}
for block_index, block in enumerate(self.blocks):
    x = block(x, **kwargs)  # Denoises video latents, hints injected via residual
```

**Decoding (src/scope/core/pipelines/wan2_1/blocks/decode.py:52):**
```python
# No frame stripping needed - latents contain ONLY video frames
video = components.vae.decode_to_pixel(block_state.latents, use_cache=True)
```

**Architecture:** Reference frames stay in a separate conditioning tensor (vace_context). The diffusion process denoises ONLY video frame latents. Reference information flows through hint injection mechanism, not through shared latent space.

## Conceptual Comparison

### Original VACE
```
Refs + Video Frames
       |
       v
  [Concatenate in latent space]
       |
       v
  Unified Latent Tensor [refs, video]
       |
       v
  [Generate noise for ALL positions]
       |
       v
  [Denoise unified tensor]
       |
       v
  [Strip reference frames]
       |
       v
  [Decode video only]
```

### Our Implementation
```
Refs                    Video Frames
  |                          |
  v                          v
vace_context             Latents [video only]
  |                          |
  v                          |
[VACE Blocks]                |
  |                          |
  v                          |
Hints --------inject-------> |
                             v
                      [Denoise video latents]
                             |
                             v
                      [Decode directly]
```

## Why This Difference Matters

### Advantages of Latent Space Separation

1. **No Frame Stripping:** Since reference frames never enter the video latent space, there's no need to strip them before decode. Cleaner pipeline flow.

2. **Fixed Chunk Sizes:** For causal/streaming generation with KV caching, maintaining consistent latent dimensions is critical. Our approach ensures each chunk always has exactly 3 latent frames (12 output frames), regardless of reference image count.

3. **Explicit Conditioning Path:** Reference information flows through a dedicated conditioning path (hints), making it architecturally clear that refs are guidance, not generation targets.

4. **Cache Compatibility:** KV cache operates on video latent positions only. No special handling needed for reference frame positions in the cache.

### Rationale for Original VACE's Approach

The original VACE uses bidirectional DiT architecture processing full sequences. Having references in the same latent tensor allows the model to jointly attend across both reference and video positions in a unified manner. This makes sense for:

- Bidirectional attention where past and future frames (including refs) are all visible
- Batch processing of complete videos rather than streaming chunks
- Simpler training setup where refs and video share the same processing path

### Why We Diverged

Our causal/streaming architecture processes video in fixed-size chunks with KV caching:

- **Chunk Independence:** Each chunk generates exactly 3 latent frames, always
- **Sequential Processing:** Can't jointly attend to "future" frames or refs in same tensor
- **Cache Alignment:** KV cache must align with consistent latent positions across chunks

Including reference frames in the latent space would break chunk size consistency and complicate cache management. By keeping refs in conditioning space, we maintain the architectural cleanliness needed for autoregressive generation.

## Hint Injection Mechanism (Shared Between Both)

Both implementations use the same hint injection pattern (this is NOT a difference):

**VACE Blocks (notes/VACE/vace/models/wan/modules/model.py:120-142):**
```python
def forward_vace(self, x, vace_context, seq_len, kwargs):
    # Embed vace_context (refs + conditioning)
    c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]

    # Process through parallel VACE transformer blocks
    for block in self.vace_blocks:
        c = block(c, **kwargs)

    # Extract hints for injection
    hints = torch.unbind(c)[:-1]
    return hints
```

**Hint Injection (modules/model.py:63-67, our attention_blocks.py:175-181):**
```python
def forward(self, x, hints, context_scale=1.0, **kwargs):
    x = super().forward(x, **kwargs)  # Standard transformer block
    if self.block_id is not None:
        x = x + hints[self.block_id] * context_scale  # Residual injection
    return x
```

The hint injection uses simple **residual addition** (not cross-attention). Zero-initialized projection layers (`self.after_proj`) in VACE blocks ensure hints start with no effect until trained.

## Secondary Difference: No Dummy Frames for R2V Mode

### Original VACE Approach

For R2V (reference-to-video) generation, when there is no source video, the original VACE uses zero tensors as temporal placeholders:

```python
# From vace/models/wan/wan_vace.py:214
if sub_src_video is None:
    src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
    src_mask[i] = torch.ones_like(src_video[i], device=device)
```

This zero tensor is then passed as `input_frames` to `vace_encode_frames()` (line 339), which encodes it for VACE context. The bidirectional DiT expects a full temporal sequence, and these zero placeholder frames fill the positions where new content will be generated.

Note: While the `frameref.py` annotator uses grayscale frames (127.5) for preprocessing/annotation purposes, the actual VACE model receives zero tensors when no source video is provided.

### Our Approach

```python
# src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py:206-213
prepared_refs_stacked = torch.cat(prepared_refs, dim=1).unsqueeze(0)
ref_latents_out = vae.encode_to_latent(prepared_refs_stacked, use_cache=False)
```

We encode reference images directly without dummy frames. This works because:

1. **Causal Architecture:** Each chunk generates independently based on fixed-size latent noise (3 frames)
2. **Conditioning Path:** References flow through vace_context/hints, not through video latent dimensions
3. **No Temporal Placeholder Needed:** Video latent temporal dimension is determined by noise shape, not by vace_context

When we tested grayscale placeholder frames in our causal setup, it produced artifacts. Direct encoding works cleanly for streaming generation.

## Implementation Files

Key files in our VACE infrastructure:

- `src/scope/core/pipelines/wan2_1/vace/models/causal_vace_model.py` - Wrapper that adds VACE to any CausalWanModel
- `src/scope/core/pipelines/wan2_1/vace/models/attention_blocks.py` - Hint generation and injection
- `src/scope/core/pipelines/wan2_1/vace/blocks/vace_encoding.py` - Encoding block (refs-only mode, no dummy frames)
- `src/scope/core/pipelines/wan2_1/vace/utils/encoding.py` - Encoding utilities (refs+conditioning concatenation)
- `src/scope/core/pipelines/wan2_1/blocks/prepare_latents.py` - Noise generation (video frames only)
- `src/scope/core/pipelines/wan2_1/blocks/decode.py` - Decode block (no frame stripping)

Original VACE reference:

- `notes/VACE/vace/models/wan/wan_vace.py` - Original pipeline (refs in latent space, frame stripping)
- `notes/VACE/vace/models/wan/modules/model.py` - Original model architecture (hint injection pattern)

---

**Summary:** The fundamental architectural difference is **where reference frames live during diffusion**. Original VACE concatenates references into the latent space being denoised (requiring frame stripping), while our implementation keeps references in a separate conditioning space (vace_context) that generates hints for the video denoising process. This separation is essential for causal/streaming generation with fixed chunk sizes and KV caching, though it diverges from the original's unified latent space approach suited for bidirectional processing.
