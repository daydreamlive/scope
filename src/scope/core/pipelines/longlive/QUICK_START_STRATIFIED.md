# Quick Start: Cache-Stratified VACE POC

## What This Does

Applies **different VACE hint strengths** to:
- **Old frames** (already in cache): Lower "refinement" scale
- **New frames** (being generated): Higher "generation" scale

This exploits the AR paradigm's explicit separation of past vs. present.

## Run the POC

```bash
# From project root
cd src/scope/core/pipelines/longlive

# Run test (requires VACE weights and depth video)
python -m scope.core.pipelines.longlive.test_stratified_vace
```

## What It Tests

| Strategy | Refinement | Generation | Expected Behavior |
|----------|-----------|------------|-------------------|
| baseline | 1.0 | 1.0 | Standard VACE (uniform) |
| strong_stratification | 0.2 | 1.0 | Creative, less anchoring |
| moderate_stratification | 0.5 | 1.0 | Balanced consistency |
| inverse_stratification | 1.0 | 0.2 | Strong drift correction |

## Output Location

```
vace_tests/stratified_poc/
├── output_baseline.mp4
├── output_strong_stratification.mp4
├── output_moderate_stratification.mp4
└── output_inverse_stratification.mp4
```

## Key Files

- **`causal_vace_model_stratified.py`**: Stratified attention blocks
- **`test_stratified_vace.py`**: Test script with 4 strategies
- **`STRATIFIED_VACE_POC.md`**: Full documentation

## How to Interpret Results

### Visual Differences to Look For

1. **Temporal Consistency**
   - Strong stratification: More variation frame-to-frame
   - Inverse stratification: Smoother, more stable

2. **Conditioning Adherence**
   - Strong stratification: Responds strongly to new depth cues
   - Inverse stratification: More conservative, anchored to past

3. **Style Drift**
   - Strong stratification: May drift from initial style
   - Inverse stratification: Maintains style consistency

## Using in Your Own Code

```python
from scope.core.pipelines.longlive import LongLivePipeline

# Standard pipeline setup
pipeline = LongLivePipeline(config, device=device)

# Generate with stratified VACE
output = pipeline(
    prompts=[{"text": "...", "weight": 100}],
    input_frames=depth_chunk,
    vace_context_scale=0.7,        # Global scale
    vace_refinement_scale=0.5,     # Scale for cached tokens
    vace_generation_scale=1.0,     # Scale for new tokens
)
```

## Troubleshooting

### "VACE checkpoint not found"
Download VACE weights to `~/.daydream-scope/models/Wan2.1-VACE-1.3B/`

### "Depth video not found"
Create or provide depth video at `vace_tests/control_frames_depth.mp4`

### Out of Memory
Reduce `max_output_frames` in test script (default 60 frames)

## Next Steps

After running the POC:

1. **Compare outputs**: Watch all 4 videos side-by-side
2. **Analyze metrics**: Check temporal consistency, FPS, latency
3. **Experiment**: Try custom refinement/generation scale combinations
4. **Extend**: Implement adaptive scaling based on content

## Why This Is Novel

Non-AR diffusion models apply uniform conditioning across all tokens. AR models with KV cache have an explicit boundary between:
- **Past** (cached, should be refined)
- **Present** (generating, should be guided)

This POC exploits that boundary for fine-grained temporal control.
