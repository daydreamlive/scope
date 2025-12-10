# Response to PR #229 Review

## Reviewer Question Summary

> IIUC `current_start_frame` is supposed to be incremented by the # of latent frames generated because the generator internally uses the # of latent frames generated thus far multiplied by the # of tokens per latent frame (eg `frame_seq_length`) when indexing into the KV cache.
>
> So, why do we need to all of sudden increment by the # of pixel frames when this is not required for other models? And why was there not a problem with T2V, but only V2V?

---

## TL;DR

The name "PrepareNextPixelFrameBlock" is somewhat misleading. The fix doesn't actually track "pixel frames" - it tracks the **effective frame stride** (consistently 3 frames) that keeps `current_start_frame` aligned across all four places where it's used. The V2V first-batch special case creates a 0→4→7→10 pattern with latent tracking, but the system needs 0→3→6→9 for temporal consistency. T2V doesn't have this issue because it doesn't have the first-batch special case.

---

## Deep Dive: The Root Cause

### Where `current_start_frame` is Actually Used

I investigated all usages of `current_start_frame` in the pipeline. It's used in **four critical places within each iteration**:

1. **PrepareVideoLatentsBlock** (line 119): `block_seed = base_seed + current_start_frame`
   - Seeds the RNG for noise generation
   - Misalignment → inconsistent noise patterns → temporal discontinuities

2. **DenoiseBlock** (line 140): `current_start = current_start_frame * frame_seq_length`
   - Passed to generator for KV cache write positioning
   - Misalignment → cache writes at wrong indices

3. **CleanKVCacheBlock** (line 120): `current_start = current_start_frame * frame_seq_length`
   - Runs generator at timestep=0 to clean cache with final denoised latents
   - Misalignment → cache cleanup at wrong positions

4. **CausalWanSelfAttention** (causal_model.py line 145): `current_start_frame = current_start // frame_seqlen`
   - Used to calculate RoPE positional encoding offsets
   - Misalignment → incorrect temporal positional embeddings

**Critical insight**: All four usages within a single iteration must use the **same** `current_start_frame` value. If they're misaligned, the system breaks.

### Empirical Proof

I implemented manual corrections in all four places (subtracting 1 when current_start_frame > 0) to convert the broken pattern (0, 4, 7, 10, 13...) to the working pattern (0, 3, 6, 9, 12...). **Result: Temporal consistency was fully restored**, proving that alignment across these four usage points is the complete root cause.

### Why V2V But Not T2V?

**T2V mode**: All chunks follow the same pattern
- Chunk 1: 3 latent frames → current_start_frame: 0 + 3 = 3
- Chunk 2: 3 latent frames → current_start_frame: 3 + 3 = 6
- Chunk 3: 3 latent frames → current_start_frame: 6 + 3 = 9
- Pattern: 0, 3, 6, 9, 12... ✓ **Aligned**

**V2V mode with PrepareNextBlock**: First batch has special case
- Chunk 1: **4** latent frames (VAE needs extra frame for temporal overlap) → current_start_frame: 0 + 4 = **4**
- Chunk 2: 3 latent frames → current_start_frame: 4 + 3 = 7
- Chunk 3: 3 latent frames → current_start_frame: 7 + 3 = 10
- Pattern: 0, **4**, 7, 10, 13... ✗ **Misaligned**

The V2V first-batch special case (requiring 4 latent frames) breaks the alignment because:
- Seeds become: base_seed+0, base_seed+**4**, base_seed+7... (should be +0, +3, +6...)
- Cache positions become: 0, **4096**, 7168... (should be 0, 3072, 6144...)
- RoPE offsets become misaligned with actual temporal progression

### Why "PrepareNextPixelFrameBlock" Works (And Why The Name Is Misleading)

From the logs:
```
DecodeBlock: VAE decoded 4 latent frames -> 13 pixel frames
PrepareNextPixelFrameBlock: incrementing current_start_frame from 0 by 3 pixel frames
```

It doesn't increment by 13! It increments by **3** - the **effective frame stride**, which is the number of new non-overlapping frames produced per iteration.

Looking at the actual code:
```python
_, _, num_output_frames, _, _ = block_state.output_video.shape
block_state.current_start_frame += num_output_frames
```

The `output_video` at this point contains only 3 frames (the non-overlapping portion), not the full 13 decoded frames. The VAE's temporal overlaps are already handled earlier in the pipeline.

**The fix works because it maintains the effective stride of 3**, giving us:
- Pattern: 0, 3, 6, 9, 12... ✓ **Aligned**
- Seeds: base_seed+0, +3, +6, +9... ✓ **Consistent**
- Cache positions: 0, 3072, 6144, 9216... ✓ **Aligned**
- RoPE offsets: 0, 3, 6, 9... ✓ **Correct**

### Why Not Required For Other Models?

Other models (longlive, krea) either:
1. Don't have the V2V first-batch special case, OR
2. Have additional buffering/recaching mechanisms (like `RecacheFramesBlock`) that mask the symptom

The reward_forcing pipeline exposed this issue because it:
- Uses V2V mode with the first-batch special case
- Lacks additional buffering that would hide the misalignment
- Has tight temporal dependencies (reward forcing mechanism) that amplify inconsistencies

### Alternative Considered

A more architectural solution would be to separate latent vs pixel frame tracking globally with something like:
- `current_start_latent_frame` for KV cache indexing
- `current_start_pixel_frame` for seed derivation and temporal tracking

However, this requires changes across all shared blocks and pipelines. The pipeline-specific fix achieves the same result with minimal impact while maintaining full compatibility with the existing architecture.

---

## Summary

The fix doesn't actually track "pixel frames" - it tracks the **effective frame stride** that keeps `current_start_frame` values aligned (0, 3, 6, 9...) across:
1. Seed derivation (PrepareVideoLatentsBlock)
2. KV cache write positions (DenoiseBlock)
3. KV cache cleanup positions (CleanKVCacheBlock)
4. RoPE positional encoding (CausalWanSelfAttention)

The V2V first-batch special case (+4 latent frames) breaks this alignment, and the fix restores it by tracking the effective stride (+3 frames consistently).

A better name might be `PrepareNextWithEffectiveStrideBlock`, but `PrepareNextPixelFrameBlock` was chosen to indicate it operates on the decoded output rather than the latent space.
