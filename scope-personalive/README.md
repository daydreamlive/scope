# PersonaLive Plugin for Daydream Scope

Real-time portrait animation plugin based on [PersonaLive](https://github.com/GVCLab/PersonaLive).

## Features

- Real-time portrait animation from reference image and driving video
- Optional TensorRT acceleration for faster inference
- Supports keyframe-based history mechanism for improved quality

## Installation

### From local directory (development)

```bash
uv run daydream-scope install -e ./scope-personalive
```

### From git repository

```bash
uv run daydream-scope install git+https://github.com/user/scope-personalive.git
```

## Usage

1. Start Daydream Scope
2. Select "PersonaLive" from the pipeline dropdown
3. Upload a reference portrait image
4. Enable video input (webcam or video file)
5. The pipeline will animate the reference portrait based on the driving video

## TensorRT Acceleration (Optional)

For faster inference, you can use TensorRT acceleration:

```bash
# Install TensorRT support
pip install daydream-scope[tensorrt]

# Convert models to TensorRT
convert-personalive-trt --model-dir ./models --height 512 --width 512
```

## Requirements

- NVIDIA GPU with CUDA support
- ~8GB VRAM
- PersonaLive model weights (downloaded automatically)

## Model Downloads

The required models will be downloaded automatically on first use:

- SD Image Variations Diffusers
- SD VAE FT MSE
- PersonaLive pretrained weights

## License

This plugin is part of Daydream Scope. See the main project for licensing information.
