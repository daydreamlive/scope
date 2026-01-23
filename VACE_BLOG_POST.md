# Training-Free Video Control for Causal Streaming Generation

Autoregressive video generation models can stream video in real-time, but they lack the control capabilities that batch models have: reference guidance, structural conditioning, selective editing. Building these from scratch would require extensive retraining. What if you could adapt existing control mechanisms instead?

This post describes an adaptation of [VACE](https://ali-vilab.github.io/VACE-Page/) (Video All-in-one Creation and Editing, Alibaba, ICCV 2025) for real-time autoregressive video generation. The adaptation enables reference-guided generation, structural conditioning, inpainting, and temporal extension in streaming contexts - using existing pretrained VACE weights without additional training.

<video src="src/blog_assets/depth/depth_comparison.mp4" controls></video>

*Real-time depth-guided video generation. Left: input video. Center: extracted depth maps. Right: generated output following structural guidance.*

---

## Background

Real-time video generation models like LongLive, FramePack, and StreamDiffusion generate video in chunks using causal attention. Each chunk attends only to itself and past frames, enabling KV caching and bounded memory usage.

VACE provides unified video control for batch-oriented diffusion models:

- **Reference-to-Video (R2V):** Style/subject guidance from reference images
- **Video-to-Video (V2V):** Structural control via depth, pose, optical flow, edges
- **Masked Video-to-Video (MV2V):** Inpainting, outpainting, temporal extension
- **Task Composition:** Arbitrary combinations of the above

However, VACE assumes bidirectional attention and processes full video sequences at once. This is incompatible with streaming generation, which requires fixed chunk sizes and causal attention patterns.

This work adapts VACE's architecture to work within these constraints while preserving its control capabilities.

---

## The Architectural Problem

### How Original VACE Handles References

VACE concatenates reference frames directly into the diffusion latent space:

```
latent = [ref_frame_1 | ref_frame_2 | video_frame_1 | video_frame_2 | ...]
```

The model processes this combined sequence with bidirectional attention, then strips the reference frames from the output after denoising.

This approach has three incompatibilities with streaming:

1. **Variable sequence lengths:** Different tasks require different numbers of reference frames, preventing fixed-size chunk processing
2. **KV cache invalidation:** Changing the reference count changes the sequence structure, invalidating cached keys/values
3. **Post-processing overhead:** Reference frames must be identified and removed after each denoising step

### The Adaptation: Separate Conditioning Space

The adaptation moves reference frames out of the diffusion latent space and into a parallel conditioning pathway:

```
Original:  latent = [refs | video]  →  denoise  →  strip refs  →  output
Adapted:   latent = [video]  +  hints_from(refs)  →  denoise  →  output
```

Reference frames are processed by separate transformer blocks (Context Blocks) that generate "hints" - additive signals injected into the main video pathway via scaled residuals.

This preserves fixed chunk sizes: video latents maintain consistent dimensions (typically 3 latent frames → 12 output frames, depending on the base pipeline), regardless of how many references are provided.

---

## Why Pretrained Weights Transfer

The publicly released VACE weights use **Context Adapter Tuning**: the base DiT is frozen, and separate Context Blocks are trained to process references and inject hints. This is the architecture we adapt.

The Context Blocks are already trained to:
- Encode reference information
- Generate hints that influence the main pathway
- Apply zero-initialized projections for gradual influence

### What Changed

| Component | Original VACE | Streaming Adaptation |
|-----------|---------------|---------------------|
| Reference input location | Concatenated into noisy latents | Separate `vace_context` tensor |
| Context Block inputs | Full sequence (refs + video) | References only |
| Hint injection target | Mixed ref+video sequence | Video-only sequence |
| Attention pattern | Bidirectional | Causal |

The Context Blocks themselves are unchanged. They process references and produce hints using the same weights. The adaptation changes where those hints are injected.

### Zero-Initialized Projections

VACE uses zero-initialized linear projections for hint injection. At initialization, hints contribute nothing. The trained weights encode how much influence to apply. These learned scaling factors remain valid in the adapted architecture.

---

## How Reference Processing Works

All VACE modes - temporal extension, structural control, inpainting, and R2V - share a common reference processing pipeline:

1. **Separate encoding:** References are VAE-encoded into a parallel `vace_context` tensor, kept separate from video latents
2. **Context Block processing:** Parallel transformer blocks process references and generate "hints"
3. **Hint injection:** Hints are added to the main video pathway via scaled residuals (`x = x + hint * scale`)
4. **Strength control:** `context_scale` (0.0–2.0) controls influence strength across all modes

The same mechanism drives depth-guided generation, first-frame extension, inpainting, and style transfer. The only difference between modes is what gets encoded as the reference.

---

## Capabilities

### Video-to-Video with Control Signals

Structural guidance from control signals processed per-chunk.

**Supported signals** (3-channel RGB from standard annotators):

| Signal | Purpose |
|--------|---------|
| Depth maps | Scene geometry |
| Pose/skeleton | Motion transfer |
| Optical flow | Motion dynamics |
| Scribble/edge | Structural guides |
| Gray | Colorization (preserve luminance) |
| Layout | Object placement via bounding boxes |

Control frames are processed per-chunk using existing VACE control encoder weights.

**Depth Control:**

<video src="src/blog_assets/depth/depth_comparison.mp4" controls></video>

*Left: input video. Center: extracted depth maps. Right: generated output following structural guidance.*

**Scribble/Edge Control:**

<video src="src/blog_assets/scribble/scribble_comparison.mp4" controls></video>

*Scribble contours extracted from video (left) provide loose structural guidance. The model interprets the edges while adding detail and style. VACE context scale: 0.9 (higher adherence to control signal).*

<video src="src/blog_assets/scribble2/scribble_comparison.mp4" controls></video>

*Same scribble input with context scale: 0.5 (lower adherence). The model takes more creative freedom while still respecting the general structure. Lower scales allow the model to deviate from the control signal, enabling more stylistic variation.*

---

### Temporal Extension

Generate video connecting to provided keyframes. Reference frames appear in the output.

**Modes:**
- `firstframe` - reference is first frame, generate continuation (useful for animating a static image)
- `lastframe` - reference is last frame, generate lead-in (useful for creating an intro to a specific endpoint)
- `firstlastframe` - two references, generate interpolation (useful for animating between storyboard keyframes)

Reference frames are encoded and placed at temporal boundaries. The model generates frames to fill the gap while maintaining coherence with anchors.

<video src="src/blog_assets/first_frame_conditioning/i2v_comparison.mp4" controls></video>

*Image-to-video generation: a single reference image (left) is used as the first frame, and the model generates a coherent video continuation (right). The FPS overlay shows real-time generation speed per chunk.*

---

### Inpainting & Outpainting

Selective region generation with masked areas regenerated while preserving the rest.

**Inpainting:**
- Static masks - same region masked every frame (e.g., fixed bounding box)
- Dynamic masks - mask varies per frame; real-time segmentation systems like SAM2 integrate well

**Outpainting:**
- Extend canvas in any direction (left, right, up, down, or combinations)
- Configurable expansion ratio

Dual-stream encoding separates reactive (to be generated) and inactive (to be preserved) regions. Each stream uses its own VAE encoder cache to prevent temporal contamination. Preserved regions maintain full quality without blending artifacts at mask boundaries.

**Character Transformation:**

<video src="src/blog_assets/inpainting/inpainting_comparison.mp4" controls></video>

*Video inpainting with YOLO-detected person masks. Original video (left), detected mask region (center), and regenerated content (right). The masked region is replaced while preserving the rest of the frame.*

**Regional LoRA Application:**

<video src="src/blog_assets/inpainting/lora_comparison.mp4" controls></video>

*Combining inpainting with LoRA style transfer. The same mask is used, but a toy soldier LoRA transforms the person into a stylized figurine while preserving the background.*

---

### Reference-to-Video (R2V) - Experimental

Reference images (1-3) guide style, subject, or character appearance. References influence generation but do not appear in output frames - think style transfer rather than keyframe interpolation.

R2V uses the same hint injection pipeline described above, but with a key difference: references provide persistent stylistic guidance across all chunks rather than anchoring specific frames.

**Note:** R2V is significantly more experimental than other capabilities. Detail preservation and reference fidelity are noticeably reduced compared to batch VACE due to causal attention constraints. The causal attention pattern and per-chunk processing fundamentally limit how well references can guide generation - R2V currently works better as coarse style guidance rather than precise subject/character transfer.

---

### Task Composition

Capabilities combine freely. The system infers mode from provided inputs:

- Multiple reference images → R2V
- Video + mask → MV2V
- Control signal → V2V
- Combinations → Composed mode

| Composition | Description |
|-------------|-------------|
| R2V + Depth | Style guidance with scene geometry |
| R2V + Inpainting | Style-consistent region replacement |
| R2V + Pose | Character animation with reference appearance |
| Extension + Outpainting | Continue video while expanding canvas |

No explicit mode parameter required.

**Layout/Trajectory Control:**

<video src="src/blog_assets/layout_control/layout_comparison.mp4" controls></video>

*Point-based subject control: a subject image is used to establish identity in the first frame (extension mode), then trajectory control guides the subject's position in subsequent chunks. The layout signal (white background with black contour) indicates where the subject should appear.*

---

## Implementation Details

### Architecture (per-chunk processing)

```
┌─────────────────────────────────────────────────────────────┐
│                     Streaming VACE                          │
├─────────────────────────────────────────────────────────────┤
│  Reference Images ──► VAE Encode ──► Context Blocks ──┐     │
│       (once)                                          │     │
│                                                       │     │
│  Video Chunk ──► VAE Encode ──► DiT Blocks ◄── hints ─┘     │
│   (each chunk)                       │                      │
│                                      ▼                      │
│                              VAE Decode ──► Output Frames   │
│                                                             │
│  [KV Cache persists across chunks]                          │
└─────────────────────────────────────────────────────────────┘
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Composition over inheritance | Pipeline-agnostic: adapts to LongLive, Krea, StreamDiffusion, etc. |
| Factory-generated block classes | Dynamically creates subclasses with hint injection support |
| Separate VAE encoder caches | Dual-stream encoding without temporal contamination |
| Zero-initialized hint projections | Safe composition with LoRA, quantization |
| Implicit mode detection | API infers mode from inputs |
| Crop-to-fill resizing | Avoids padding artifacts |
| Cached hint computation | Reference hints computed once, reused across chunks |

### Pipeline Compatibility

All autoregressive pipelines in the codebase support VACE via the `VACEEnabledPipeline` mixin:

| Base Pipeline | Status |
|--------------|--------|
| LongLive | Full support |
| StreamDiffusion V2 | Full support |
| MemFlow | Full support |
| Krea Realtime Video | Full support |
| Reward Forcing | Full support |

---

## Performance

*Benchmarks measured on single NVIDIA RTX 5090 32GB at 368×640 resolution (portrait 480p), 1.3B parameter model with SageAttention enabled.*

### Latency (per chunk, 12 frames)

| Configuration | Avg Latency | Avg Throughput | Peak Throughput |
|--------------|-------------|----------------|-----------------|
| LongLive + Depth Control | 570ms | 20.6 fps | 22.5 fps |
| LongLive + Scribble Control | 570ms | 20.6 fps | 22.7 fps |
| LongLive + Inpainting | 570ms | 20.7 fps | 22.6 fps |
| LongLive + Layout/Trajectory | 700ms | 17.5 fps | 18.2 fps |
| LongLive + Extension (I2V) | 400ms | 27.7 fps | 33.1 fps |
| LongLive + Inpainting + LoRA | 900ms | 12.8 fps | 13.4 fps |

*First chunk of each generation has higher latency (~630ms) due to KV cache warmup.*

### Memory

| Configuration | VRAM |
|--------------|------|
| 1.3B model, 480p | ~12 GB |
| 14B model, 720p | ~28 GB |

## Comparison to Alternatives

The primary alternative for real-time controlled video generation is **MotionStream**, a fully distilled model with built-in trajectory control. MotionStream is purpose-built for a single control modality and achieves higher quality for that specific use case. However, it requires full model retraining for each control type.

This VACE adaptation trades some quality for versatility: a single set of pretrained weights enables depth control, scribble guidance, inpainting, layout control, and arbitrary combinations - without retraining. The approach is more extensible to new control types as the community develops them for batch VACE.

---

## Limitations & Known Issues

### Quality Considerations

- **Temporal coherence:** Can degrade over extended generations (100+ frames) without re-anchoring or keyframe injection - this is largely a consequence of autoregression in general
- **Control signal variance:** Some signals (depth, scribble, layout) work reliably, while others need more tuning
- **First+last frame extension in combination:** Reduced utility when compared to batch paradigm due to small chunk sizes in streaming contexts

### Known Failure Cases

**Reference-to-Video (R2V):** This is the most problematic capability in the streaming adaptation. Detail preservation and reference fidelity are severely degraded compared to batch VACE. The causal attention pattern and per-chunk processing fundamentally limit how well references can guide generation. R2V currently works better as coarse style guidance rather than precise subject/character transfer. Further architectural work is needed to approach batch-quality R2V in streaming contexts.

### Resource Requirements

- 14B models require VRAM optimization for consumer hardware
- Multi-GPU inference adds latency overhead due to communication costs

### Coverage Gaps

The batch VACE ecosystem has accumulated extensive community-driven examples and techniques over months of use—various control signal combinations, preprocessing pipelines, and creative workflows. Many remain unexplored in the streaming context.

---


## Summary

By moving reference frames from the diffusion latent space into a parallel conditioning pathway, this adaptation preserves the fixed chunk sizes and KV caching that autoregressive models require—while reusing existing VACE weights directly.

**Key contributions:**

1. **Zero-shot weight transfer:** Existing VACE weights work directly in streaming contexts
2. **Maintained capabilities:** Structural control, masked generation, and temporal extension all function in real-time
3. **Pipeline agnostic:** The composition-based design adapts to different autoregressive base models
4. **Practical performance:** 20+ fps generation with control on consumer hardware

<video src="src/blog_assets/scribble/scribble_comparison.mp4" controls></video>

*All demos generated in real-time with FPS overlay showing actual generation speed per chunk. Try it yourself in [Daydream Scope](https://github.com/daydreamlive/scope).*
