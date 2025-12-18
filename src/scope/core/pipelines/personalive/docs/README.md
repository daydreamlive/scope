# PersonaLive Pipeline

PersonaLive is a real-time portrait animation pipeline that animates a reference portrait image using driving video frames.

## Overview

PersonaLive takes:
1. A **reference image** - A portrait photograph of the person to animate
2. **Driving video frames** - Video frames that provide the motion/expression to transfer

And outputs animated video frames of the reference person following the expressions and movements from the driving video.

## Installation

### 1. Download Models

```bash
# Download PersonaLive models
download_models --pipeline personalive
```

This downloads from [huggingface.co/huaichang/PersonaLive](https://huggingface.co/huaichang/PersonaLive), which includes:
- PersonaLive custom weights (denoising_unet, reference_unet, motion_encoder, etc.)
- sd-image-variations-diffusers base model
- sd-vae-ft-mse VAE

### 2. Install Dependencies

Make sure you have the required dependencies installed:
- `mediapipe>=0.10.11` - For face detection and keypoint extraction
- `decord>=0.6.0` - For video decoding
- `xformers>=0.0.28` - For memory-efficient attention

## Usage

### API Usage

1. **Load the pipeline**:

```bash
POST /api/v1/pipeline/load
Content-Type: application/json

{
  "pipeline_id": "personalive",
  "load_params": {
    "height": 512,
    "width": 512,
    "seed": 42
  }
}
```

2. **Upload reference image**:

```bash
POST /api/v1/personalive/reference
Content-Type: image/jpeg

<raw image bytes>
```

3. **Start WebRTC stream**:

Connect via WebRTC and send driving video frames. The output will be animated frames of the reference portrait.

### Workflow

```
1. Load PersonaLive pipeline
2. Upload reference portrait image
3. Connect WebRTC stream with camera/video input
4. Receive animated output frames
```

## Technical Details

### Architecture

PersonaLive uses a multi-component architecture:

- **Reference UNet (2D)**: Extracts features from the reference image
- **Denoising UNet (3D)**: Generates animated frames with temporal consistency
- **Motion Encoder**: Extracts motion features from driving video faces
- **Pose Encoder (MotionExtractor)**: Extracts keypoints from faces
- **Pose Guider**: Converts keypoints to conditioning features
- **VAE**: Encodes/decodes latent representations
- **Image Encoder (CLIP)**: Extracts semantic features from reference image

### Processing Flow

1. **Reference Fusion** (once at start):
   - Encode reference image with CLIP
   - Encode reference image with VAE
   - Run Reference UNet to cache attention features
   - Extract reference face keypoints

2. **Frame Processing** (for each driving frame batch):
   - Extract face keypoints from driving frames
   - Compute motion features from cropped face regions
   - Generate pose condition embeddings
   - Denoise latents with temporal attention
   - Decode output frames from latents

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `height` | 512 | Output video height |
| `width` | 512 | Output video width |
| `seed` | 42 | Random seed for reproducibility |
| `temporal_window_size` | 4 | Number of frames processed together |
| `temporal_adaptive_step` | 4 | Temporal denoising step |
| `num_inference_steps` | 4 | Denoising steps (typically 4 for real-time) |

## References

- [PersonaLive Paper](https://arxiv.org/abs/2506.09887)
- [GitHub Repository](https://github.com/GVCLab/PersonaLive)
- [HuggingFace Models](https://huggingface.co/huaichang/PersonaLive)

## Credits

Based on the PersonaLive implementation by GVCLab.

