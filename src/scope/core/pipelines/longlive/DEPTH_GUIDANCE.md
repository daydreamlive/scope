# Deprecated: See CONDITIONING_GUIDANCE.md

This file has been superseded by `CONDITIONING_GUIDANCE.md` which describes the new, simplified VACE conditioning API.

## What Changed

The new API is simpler and more flexible:

**Old API (Deprecated):**
```python
# Had to specify guidance_mode explicitly
output = pipeline(
    guidance_mode="depth",  # Explicit mode
    vace_input=depth_chunk,
)

# Reference images were cached internally
output = pipeline(
    guidance_mode="r2v",
    ref_images=ref_images,  # Only needed on first chunk
)

# Could NOT combine both
```

**New API (Current):**
```python
# Mode is implicit - just provide what you want
output = pipeline(
    input_frames=conditioning_chunk,  # Any conditioning type (depth, flow, pose, etc.)
    input_masks=mask_chunk,           # Optional: spatial control (defaults to ones)
)

# Reference images sent every chunk (application manages reuse)
output = pipeline(
    ref_images=ref_images,  # Send each chunk
)

# Can combine both!
output = pipeline(
    ref_images=ref_images,            # Style/character
    input_frames=conditioning_chunk,  # Structure (depth, flow, etc.)
    input_masks=mask_chunk,           # Optional: spatial control
)
```

## Benefits

1. **Simpler**: No explicit `guidance_mode` parameter
2. **More flexible**: Combine reference images + conditioning
3. **Clearer**: Mode is implicit based on what's provided
4. **More control**: Application layer manages input reuse
5. **Works for any conditioning**: depth, flow, pose, scribble, etc.
6. **Matches VACE reference**: Parameter names match notes/VACE exactly

## Migration Guide

See `CONDITIONING_GUIDANCE.md` for complete documentation.

Quick migration:
- Replace `vace_input` with `input_frames`
- Optionally add `input_masks` for spatial control (defaults to ones if not provided)
- Remove `guidance_mode` parameter - mode is now implicit
- Send `ref_images` on every chunk instead of just first chunk
- Can now combine `ref_images` + `input_frames` for best of both worlds
