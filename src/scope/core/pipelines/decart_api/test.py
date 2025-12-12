import os
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf

try:
    from ..video import load_video
    VIDEO_LOADING_AVAILABLE = True
except Exception as e:
    # Try alternative video loading with imageio
    try:
        import imageio
        VIDEO_LOADING_AVAILABLE = True
        USE_IMAGEIO = True
    except ImportError:
        print(f"Warning: Video loading not available: {e}")
        print("Will skip video loading test")
        VIDEO_LOADING_AVAILABLE = False
        USE_IMAGEIO = False
else:
    USE_IMAGEIO = False

from .pipeline import DecartApiPipeline

# Check for API key
api_key = os.getenv("DECART_API_KEY")
if not api_key:
    raise ValueError(
        "DECART_API_KEY environment variable is required. "
        "Set it before running this test."
    )

# Create config - using same resolution as streamdiffusionv2 test
config = OmegaConf.create(
    {
        "height": 480,
        "width": 832,
        "seed": 42,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize pipeline
print("Initializing DecartApiPipeline...")
pipeline = DecartApiPipeline(
    config,
    device=device,
    dtype=torch.bfloat16,
)
print("Pipeline initialized successfully!")

if not VIDEO_LOADING_AVAILABLE:
    print("\n=== Skipping video loading test (video loading not available) ===")
    print("Please install FFmpeg and torchcodec to test with video input.")
    exit(0)

# Load the same input video as streamdiffusionv2 test
print("\n=== Loading input video ===")
video_path = Path(__file__).parent.parent / "streamdiffusionv2" / "assets" / "original.mp4"
if not video_path.exists():
    raise FileNotFoundError(
        f"Input video not found at {video_path}. "
        "Please ensure the streamdiffusionv2/assets/original.mp4 file exists."
    )

if USE_IMAGEIO:
    # Use imageio as fallback
    import imageio
    from PIL import Image

    print("Using imageio to load video...")
    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        # Resize frame
        img = Image.fromarray(frame)
        img = img.resize((config.width, config.height), Image.Resampling.LANCZOS)
        frame_resized = torch.from_numpy(np.array(img)).float()
        frames.append(frame_resized)
    reader.close()

    # Stack frames: T H W C -> C T H W
    input_video_cthw = torch.stack(frames, dim=0)  # T H W C
    input_video_cthw = rearrange(input_video_cthw, "T H W C -> C T H W")

    # Convert to BCTHW and ensure [0, 255] range
    input_video_bcthw = rearrange(input_video_cthw, "C T H W -> 1 C T H W")
    input_video_bcthw = input_video_bcthw.clamp(0, 255)
else:
    # Load video - load_video returns CTHW format, normalized to [-1, 1]
    # We need to convert to [0, 255] range and BCTHW format for the pipeline
    input_video_cthw = load_video(
        str(video_path),
        resize_hw=(config.height, config.width),
        normalize=True,  # Returns [-1, 1] range
    )

    # Convert from CTHW to BCTHW and denormalize to [0, 255]
    input_video_bcthw = rearrange(input_video_cthw, "C T H W -> 1 C T H W")
    # Denormalize from [-1, 1] to [0, 255]
    input_video_bcthw = ((input_video_bcthw + 1.0) / 2.0 * 255.0).clamp(0, 255)

_, _, num_input_frames, _, _ = input_video_bcthw.shape
print(f"Input video loaded: {num_input_frames} frames, shape={input_video_bcthw.shape}")

# Process video in chunks (decart_api processes one frame at a time)
chunk_size = 1  # Decart API processes one frame at a time
prompts = [{"text": "a bear is walking on the grass", "weight": 100}]

outputs = []
latency_measures = []
fps_measures = []
total_input_frames = 0
total_output_frames = 0

print(f"\n=== Processing {num_input_frames} frames in chunks of {chunk_size} ===")
for start_idx in range(0, num_input_frames, chunk_size):
    end_idx = min(start_idx + chunk_size, num_input_frames)
    chunk = input_video_bcthw[:, :, start_idx:end_idx]

    start = time.time()
    # Process chunk through pipeline
    output = pipeline(video=chunk, prompts=prompts)
    latency = time.time() - start

    num_output_frames, _, _, _ = output.shape
    fps = num_output_frames / latency if latency > 0 else 0

    input_frames_in_chunk = end_idx - start_idx
    total_input_frames += input_frames_in_chunk
    total_output_frames += num_output_frames

    print(
        f"Chunk [{start_idx}:{end_idx}]: "
        f"Input {input_frames_in_chunk} frames -> "
        f"Output {num_output_frames} frames, "
        f"latency={latency:.3f}s, "
        f"fps={fps:.2f}"
    )

    latency_measures.append(latency)
    fps_measures.append(fps)
    outputs.append(output.detach().cpu())

# Concatenate all outputs
output_video = torch.concat(outputs)
print(f"\n=== Frame Count Comparison ===")
print(f"Total input frames: {total_input_frames}")
print(f"Total output frames: {total_output_frames}")
print(f"Frame ratio (output/input): {total_output_frames / total_input_frames:.3f}")

if abs(total_output_frames - total_input_frames) <= 1:
    print("✓ Output frame count matches input frame count!")
else:
    print(
        f"⚠ Frame count difference: {abs(total_output_frames - total_input_frames)} frames"
    )

print(f"\nOutput video shape: {output_video.shape}")

# Export to video
output_path = Path(__file__).parent / "output.mp4"
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, output_path, fps=16)
print(f"Output video saved to: {output_path}")

# Print statistics
print("\n=== Performance Statistics ===")
if latency_measures:
    print(
        f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.3f}s, "
        f"Max: {max(latency_measures):.3f}s, "
        f"Min: {min(latency_measures):.3f}s"
    )
if fps_measures:
    print(
        f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, "
        f"Min: {min(fps_measures):.2f}"
    )

print("\n=== Test completed successfully! ===")
