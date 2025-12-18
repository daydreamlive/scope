# VACE (Video All-in-One Creation and Editing)

VACE adds reference image conditioning to LongLive and StreamDiffusionV2 pipelines.

## Features

The web interface supports:
- **Reference Image Conditioning**: Upload reference images to guide video generation

The pipelines are also capable of:
- **Image Guidance with Depth Maps**: Structural control using depth information
- **Inpainting**: Masked video-to-video generation

## Usage

### Web Interface

1. **Load Pipeline**: Select LongLive or StreamDiffusionV2
2. **Upload Reference Images**: Use the image manager in the controls panel
3. **Adjust VACE Scale**: Control conditioning strength (0.0-2.0, default 1.0)
4. **Generate**: Start streaming with reference image guidance

### Advanced Usage (Python API)

For depth guidance and inpainting examples, see:
- [`src/scope/core/pipelines/longlive/test_vace.py`](../src/scope/core/pipelines/longlive/test_vace.py)

This test script demonstrates:
- R2V (Reference-to-Video) generation
- Depth guidance using depth maps
- Inpainting with masks
- Combining multiple modes (R2V + Depth, R2V + Inpainting, etc.)

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_images` | `list[str]` | `None` | List of reference image paths |
| `vace_context_scale` | `float` | `1.0` | Conditioning strength (0.0-2.0) |

Higher `vace_context_scale` values make reference images more influential. Lower values allow more creative freedom while maintaining general guidance.

## Model Requirements

VACE requires the `Wan2.1-VACE-1.3B` model, which is automatically downloaded when you download LongLive or StreamDiffusionV2 models.
