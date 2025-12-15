"""
Test script for LongLive depth-guided video generation.

This script demonstrates:
1. Loading a video and generating depth maps using DepthAnything
2. Generating video with depth guidance using LongLive pipeline via standard VACE path
3. Comparing depth-guided vs non-guided generation

Follows original VACE architecture (notes/VACE/vace/models/wan/wan_vace.py):
- Depth maps treated as input_frames (3-channel RGB from annotators)
- Standard encoding: vace_encode_frames (with masks=ones) -> vace_encode_masks -> vace_latent
- masks = ones (all white masks, goes through masking path), ref_images = None

Requirements:
- Install DepthAnything: pip install depth-anything-v2
- Or use any depth estimation model (MiDaS, etc.)

Usage:
    python -m scope.core.pipelines.longlive.test_depth_guidance
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline
from .vace_utils import extract_depth_chunk, preprocess_depth_frames


def load_video_frames(video_path, target_height=480, target_width=832, max_frames=None):
    """Load video frames from file."""
    print(f"load_video_frames: Loading video from {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            frame = cv2.resize(frame, (target_width, target_height))

        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()

    print(
        f"load_video_frames: Loaded {len(frames)} frames at {target_height}x{target_width}"
    )
    return np.array(frames)


def generate_depth_maps_depthanything(frames, model_size="small"):
    """
    Generate depth maps using DepthAnything.

    Args:
        frames: Video frames [F, H, W, C] in RGB format (0-255)
        model_size: Model size ("small", "base", or "large")

    Returns:
        Depth maps [F, H, W] normalized to [0, 1]
    """
    print(
        f"generate_depth_maps_depthanything: Generating depth maps using DepthAnything-{model_size}"
    )

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as err:
        raise ImportError(
            "DepthAnything not installed. Install with: pip install depth-anything-v2"
        ) from err

    # Model configuration
    model_configs = {
        "small": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "base": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "large": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = model_configs[model_size]

    # Initialize model
    depth_model = DepthAnythingV2(**config)

    # Load pretrained weights (you'll need to download these)
    # For now, this is a placeholder - you need to provide the checkpoint path
    # depth_model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{model_size}.pth"))

    depth_model = depth_model.to(device).eval()

    depth_maps = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(
                    f"generate_depth_maps_depthanything: Processing frame {i+1}/{len(frames)}"
                )

            # DepthAnything expects HWC RGB format
            depth = depth_model.infer_image(frame)

            # Normalize to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

            depth_maps.append(depth)

    print(f"generate_depth_maps_depthanything: Generated {len(depth_maps)} depth maps")
    return np.array(depth_maps)


def generate_depth_maps_simple(frames):
    """
    Generate simple depth maps using Canny edge detection (for testing).
    This is a placeholder for when DepthAnything is not available.

    Args:
        frames: Video frames [F, H, W, C] in RGB format (0-255)

    Returns:
        Depth maps [F, H, W] normalized to [0, 1]
    """
    print("generate_depth_maps_simple: Generating simple depth maps using Canny edges")

    depth_maps = []

    for _i, frame in enumerate(frames):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Invert (edges are dark, background is light)
        depth = 255 - edges

        # Apply some blur for smoothness
        depth = cv2.GaussianBlur(depth, (9, 9), 0)

        # Normalize to [0, 1]
        depth = depth.astype(np.float32) / 255.0

        depth_maps.append(depth)

    print(f"generate_depth_maps_simple: Generated {len(depth_maps)} depth maps")
    return np.array(depth_maps)


def main():
    # Configuration
    output_dir = Path(__file__).parent / "vace_tests" / "depth_guidance"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Pipeline configuration
    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            # Skip LoRA for this test to avoid rank mismatch issues
            # "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "vace_path": str(
                get_model_file_path(
                    "Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"
                )
            ),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
            "height": 480,
            "width": 832,
        }
    )

    # Override vace_in_dim for depth mode
    config.model_config.base_model_kwargs = config.model_config.base_model_kwargs or {}
    config.model_config.base_model_kwargs["vace_in_dim"] = (
        96  # Use 96 to load pretrained R2V weights
    )

    device = torch.device("cuda")
    pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)

    # Test parameters
    prompt_text = "A cat"
    num_output_frames = 36  # 3 chunks * 12 frames

    # Load depth maps from existing video
    depth_video_path = Path(__file__).parent / "vace_tests" / "control_frames_depth.mp4"
    if not depth_video_path.exists():
        raise FileNotFoundError(f"Depth video not found at {depth_video_path}")

    print(f"\nLoading depth maps from {depth_video_path}")
    depth_frames_rgb = load_video_frames(
        depth_video_path,
        target_height=config.height,
        target_width=config.width,
        max_frames=num_output_frames,
    )  # [F, H, W, C]

    # Convert RGB to grayscale (depth maps are single-channel)
    # Take the first channel since depth videos are typically grayscale saved as RGB
    depth_frames_np = depth_frames_rgb[:, :, :, 0]  # [F, H, W]
    # Normalize to [0, 1] range
    depth_frames_np = depth_frames_np.astype(np.float32) / 255.0

    # Preprocess depth frames
    depth_frames_tensor = torch.from_numpy(depth_frames_np).float()  # [F, H, W]
    depth_video = preprocess_depth_frames(
        depth_frames_tensor,
        config.height,
        config.width,
        device,
    )  # [1, 1, F, H, W]

    print(f"\nDepth video shape: {depth_video.shape}")

    # Generate video with depth guidance
    print("\n=== Generating video with depth guidance ===")
    outputs_depth = []
    latency_measures_depth = []
    fps_measures_depth = []

    num_frames_generated = 0
    chunk_index = 0
    frames_per_chunk = 12  # 3 latent frames * 4 temporal upsample

    while num_frames_generated < num_output_frames:
        start = time.time()

        # Check if we have enough depth frames for this chunk
        depth_frames_available = depth_video.shape[2]
        depth_frames_needed = chunk_index * frames_per_chunk + frames_per_chunk
        if depth_frames_needed > depth_frames_available:
            print(
                f"\nStopping: Not enough depth frames for chunk {chunk_index}. "
                f"Need {depth_frames_needed}, have {depth_frames_available}"
            )
            break

        # Extract depth chunk for this generation step
        depth_chunk = extract_depth_chunk(
            depth_video,
            chunk_index,
            frames_per_chunk,
        )

        print(f"\nChunk {chunk_index}: depth_chunk shape = {depth_chunk.shape}")

        # Generate with depth guidance (using standard VACE path)
        output = pipeline(
            prompts=[{"text": prompt_text, "weight": 100}],
            input_frames=depth_chunk,
            vace_context_scale=0.7,
        )

        num_output_frames_chunk, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames_chunk / latency

        print(
            f"Chunk {chunk_index}: Generated {num_output_frames_chunk} frames "
            f"latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures_depth.append(latency)
        fps_measures_depth.append(fps)
        num_frames_generated += num_output_frames_chunk
        chunk_index += 1
        outputs_depth.append(output.detach().cpu())

    # Save depth-guided video
    output_video_depth = torch.concat(outputs_depth)
    print(f"\nDepth-guided output shape: {output_video_depth.shape}")
    output_video_depth_np = output_video_depth.contiguous().numpy()
    export_to_video(
        output_video_depth_np,
        output_dir / "output_depth_guided.mp4",
        fps=16,
    )

    print("\n=== Depth-Guided Generation Statistics ===")
    print(
        f"Latency - Avg: {sum(latency_measures_depth) / len(latency_measures_depth):.2f}s, "
        f"Max: {max(latency_measures_depth):.2f}s, "
        f"Min: {min(latency_measures_depth):.2f}s"
    )
    print(
        f"FPS - Avg: {sum(fps_measures_depth) / len(fps_measures_depth):.2f}, "
        f"Max: {max(fps_measures_depth):.2f}, "
        f"Min: {min(fps_measures_depth):.2f}"
    )

    # Save depth visualization
    print("\nSaving depth visualization...")
    depth_vis = (depth_frames_np[:num_frames_generated] * 255).astype(np.uint8)
    depth_vis_rgb = np.stack(
        [depth_vis, depth_vis, depth_vis], axis=-1
    )  # Convert to RGB
    export_to_video(
        depth_vis_rgb,
        output_dir / "depth_maps.mp4",
        fps=16,
    )

    print(f"\nOutputs saved to {output_dir}/")
    print("- output_depth_guided.mp4: Generated video with depth guidance")
    print("- depth_maps.mp4: Depth map visualization")


if __name__ == "__main__":
    main()
