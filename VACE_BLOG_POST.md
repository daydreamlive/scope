# Training-Free Video Control for Causal Streaming Generation

Real-time video generation is here, but controlling it—style references, structural guidance, selective editing—has meant training separate models for each capability. What if you could have all of them at once, without training anything?

This post describes an adaptation of [VACE](https://ali-vilab.github.io/VACE-Page/) (Video All-in-one Creation and Editing, Alibaba, ICCV 2025) for real-time autoregressive video generation. The adaptation enables reference-guided generation, structural conditioning, inpainting, and temporal extension in streaming contexts—using existing pretrained VACE weights without additional training.

<insert hero demo here: real-time generation with style reference + depth control>

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

Reference frames are processed by separate transformer blocks (Context Blocks) that generate "hints"—additive signals injected into the main video pathway via scaled residuals.

This preserves fixed chunk sizes: video latents maintain consistent dimensions (typically 3 latent frames → 12 output frames, depending on the base pipeline), regardless of how many references are provided.

<insert architecture diagram here: original VACE (refs in latent) vs adapted (refs in conditioning space)>

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

<insert control signal demo here: webcam → depth extraction → stylized output>

---

### Temporal Extension

Generate video connecting to provided keyframes. Reference frames appear in the output.

**Modes:**
- `firstframe` — reference is first frame, generate continuation (useful for animating a static image)
- `lastframe` — reference is last frame, generate lead-in (useful for creating an intro to a specific endpoint)
- `firstlastframe` — two references, generate interpolation (useful for animating between storyboard keyframes)

**Clip-based variants:**
- `firstclip` / `lastclip` / `firstlastclip` — video segments as anchors instead of single frames

Reference frames are encoded and placed at temporal boundaries. The model generates frames to fill the gap while maintaining coherence with anchors.

<insert extension demo here: two keyframes → interpolated video>

---

### Inpainting & Outpainting

Selective region generation with masked areas regenerated while preserving the rest.

**Inpainting:**
- Static masks — same region masked every frame (e.g., fixed bounding box)
- Dynamic tracking — mask follows a subject across frames using SAM2 segmentation, bounding box tracking, or label/caption-based detection

**Outpainting:**
- Extend canvas in any direction (left, right, up, down, or combinations)
- Configurable expansion ratio

Dual-stream encoding separates reactive (to be generated) and inactive (to be preserved) regions via VACE's "concept decoupling" mechanism. Preserved regions maintain full quality without blending artifacts at mask boundaries.

<insert inpainting demo here: background replacement with preserved subject>

---

### Reference-to-Video (R2V)

Reference images (1-3) guide style, subject, or character appearance. Unlike the modes above, this capability diverges more significantly from original VACE's behavior in the streaming context. References influence generation but do not appear in output frames—think style transfer rather than keyframe interpolation.

- References are VAE-encoded separately from video latents
- Context Blocks process references and generate persistent hints
- Hints influence all chunks via scaled residual injection
- `context_scale` (0.0–2.0) controls influence strength

<insert R2V demo here: same prompt with different style references>

<insert context scale demo here: same reference at 0.5, 1.0, 1.5, 2.0>

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

<insert composition demo here: style reference + depth control>

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
| Composition over inheritance | Pipeline-agnostic: adapts to LongLive, FramePack, StreamDiffusion, etc. |
| Factory-generated block classes | Dynamically wraps attention blocks with hint injection |
| Separate VAE encoder caches | Dual-stream encoding without temporal contamination |
| Zero-initialized hint projections | Safe composition with LoRA, quantization |
| Implicit mode detection | API infers mode from inputs |
| Crop-to-fill resizing | Avoids padding artifacts |
| Cached hint computation | Reference hints computed once, reused across chunks |

### Pipeline Compatibility

| Base Pipeline | Status |
|--------------|--------|
| LongLive | Full support |
| FramePack | Full support |
| StreamDiffusion V2 | In progress |
| MemFlow | Experimental |

---

## Comparison

| Aspect | Original VACE | Streaming Adaptation |
|--------|---------------|---------------------|
| Base architecture | Bidirectional DiT | Causal transformer |
| Attention pattern | Full sequence | Causal |
| Reference location | In diffusion latents | Separate conditioning space |
| Sequence handling | Full video batch | Fixed-size chunks + KV cache |
| Streaming | No | Yes |
| KV cache compatible | No | Yes |
| Additional training | N/A (pretrained weights used as-is) | None (same weights, no modification) |
| Frame stripping | Required | Not needed |
| Memory scaling | O(n²) with length | O(1) per chunk |

---

## Performance

<insert benchmark data>

*Benchmarks measured on single NVIDIA A100 80GB, default inference settings, 25 denoising steps unless noted.*

### Latency (per chunk)

| Configuration | Latency | Throughput |
|--------------|---------|------------|
| LongLive + R2V | <X ms> | <Y fps> |
| LongLive + Depth | <X ms> | <Y fps> |
| LongLive + R2V + Depth | <X ms> | <Y fps> |

### Memory

| Configuration | VRAM |
|--------------|------|
| 1.3B model, 480p | <X GB> |
| 14B model, 720p | <X GB> |

### Quality

<insert quality comparison: streaming vs batch VACE>

- Reference fidelity: <notes>
- Temporal coherence: <notes>
- Control adherence: <notes>

---

## Limitations

**Quality considerations:**
- Temporal coherence can degrade over extended generations (>100 chunks) without re-anchoring
- R2V detail preservation slightly reduced compared to batch VACE due to causal attention constraints
- Some control signals (optical flow) more sensitive to chunk boundaries

**Resource requirements:**
- 14B models require VRAM optimization
- Multi-GPU inference adds latency overhead

**Coverage gaps:**
- Clip-based extension less tested in streaming context
- Some composition combinations need additional validation

---

## Usage

This adaptation is available in Daydream Scope's LongLive pipeline. Basic usage:

```python
# R2V with style reference
pipeline.generate(
    prompt="a woman walking through a forest",
    vace_reference_images=["style_ref.png"],
    vace_context_scale=1.0,
)

# V2V with depth control
pipeline.generate(
    prompt="a cyberpunk cityscape",
    vace_video=depth_map_video,
)

# Combined: style reference + depth control
pipeline.generate(
    prompt="a woman walking through a forest",
    vace_reference_images=["style_ref.png"],
    vace_video=depth_map_video,
    vace_context_scale=1.0,
)
```

Mode is inferred from which inputs are provided. See the API documentation for full parameter reference.

---

## Summary

This adaptation brings VACE's control capabilities to real-time autoregressive video generation by moving reference frames from the diffusion latent space to a parallel conditioning pathway. This preserves fixed chunk sizes for KV caching while enabling the same reference-guided, conditioned, and masked generation that VACE provides in batch mode.

Existing VACE weights transfer without modification due to the Context Adapter architecture used in the public release.

<insert closing demo here>
