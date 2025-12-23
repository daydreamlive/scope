"""
Test script for VACE R2V using StreamDiffusionV2 pipeline.

This uses pipeline.py which imports causal_vace_model.py
"""

import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import StreamDiffusionV2Pipeline

print("\n" + "=" * 80)
print("  TESTING: StreamDiffusionV2 R2V WITH VACE")
print("=" * 80 + "\n")

# Check if VACE model is available
vace_model_path = Path.home() / ".daydream-scope" / "models" / "Wan2.1-VACE-1.3B"
vace_checkpoint = vace_model_path / "diffusion_pytorch_model.safetensors"

if not vace_checkpoint.exists():
    print(f"VACE checkpoint not found at {vace_checkpoint}")
    print("Please download VACE weights to ~/.daydream-scope/models/Wan2.1-VACE-1.3B/")
    print("Falling back to standard StreamDiffusionV2 model (no VACE conditioning)")
    vace_path = None
else:
    vace_path = str(vace_checkpoint)
    print(f"Using VACE model from {vace_path}")

config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")
        ),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "vace_path": vace_path,
        "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
        "height": 512,
        "width": 512,
    }
)

device = torch.device("cuda")
pipeline = StreamDiffusionV2Pipeline(
    config,
    device=device,
    dtype=torch.bfloat16,
)

# Test prompt
prompt_text = ""

# Example reference images (provide your own paths)
# These images will condition the video generation
ref_images = [
    "frontend/public/assets/example.png",
    # "path/to/reference_image_1.png",
    # "path/to/reference_image_2.jpg",
]

# If no reference images provided, try to find example.png in project root
if not ref_images:
    example_img = Path(__file__).parent.parent.parent.parent.parent / "example.png"
    if example_img.exists():
        ref_images = [str(example_img)]
        print(f"Using example image: {example_img}")
    else:
        print("No reference images provided. Running without VACE conditioning.")
        print(
            "To use VACE conditioning, provide reference image paths in the ref_images list."
        )

# Generate video
outputs = []
latency_measures = []
fps_measures = []

num_frames = 0
max_output_frames = 45
is_first_chunk = True
while num_frames < max_output_frames:
    start = time.time()

    prompts = [{"text": prompt_text, "weight": 100}]

    # Generate with VACE conditioning if reference images provided
    # Only send ref_images on the first chunk
    # Mode is inferred from absence of video parameter (text mode)
    kwargs = {"prompts": prompts}
    if is_first_chunk and ref_images and vace_path is not None:
        kwargs["vace_ref_images"] = ref_images
        kwargs["vace_context_scale"] = 1.0

    output = pipeline(**kwargs)
    is_first_chunk = False

    num_output_frames, _, _, _ = output.shape
    latency = time.time() - start
    fps = num_output_frames / latency

    print(
        f"Pipeline generated {num_output_frames} frames latency={latency:.2f}s fps={fps:.2f}"
    )

    latency_measures.append(latency)
    fps_measures.append(fps)
    num_frames += num_output_frames
    outputs.append(output.detach().cpu())

# Concatenate all THWC tensors
output_video = torch.concat(outputs)
has_nan = torch.isnan(output_video).any().item()
has_inf = torch.isinf(output_video).any().item()
print(
    f"Final output: shape={output_video.shape}, nan={has_nan}, inf={has_inf}, range=[{output_video.min().item():.2f},{output_video.max().item():.2f}]"
)

# Export video - output is already in [0, 1] range from postprocess_chunk
output_path = (
    Path(__file__).parent / "vace_tests" / "r2v" / "output_streamdiffusionv2_r2v.mp4"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, output_path, fps=16)
print(f"Saved video to {output_path}")

# Print statistics
print("\n=== Performance Statistics ===")
print(
    f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
    f"Max: {max(latency_measures):.2f}s, "
    f"Min: {min(latency_measures):.2f}s"
)
print(
    f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
    f"Max: {max(fps_measures):.2f}, "
    f"Min: {min(fps_measures):.2f}"
)

print("\n" + "=" * 80)
print("  COMPLETED: StreamDiffusionV2 R2V WITH VACE")
print("=" * 80 + "\n")
print(f"Output saved to: {output_path}")
if ref_images:
    print(f"Used {len(ref_images)} reference image(s) for conditioning")
else:
    print("No reference images used (standard T2V generation)")
