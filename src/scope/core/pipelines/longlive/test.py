"""Quick test for LongLivePipeline with new MultiModePipeline architecture.

Tests both text-to-video and video-to-video modes with a short output.
"""

import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

print("test.py: Initializing LongLivePipeline with MultiModePipeline architecture...")

config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
        ),
        "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
        "height": 320,
        "width": 576,
    }
)

device = torch.device("cuda")
pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)

print(f"test.py: Pipeline initialized. Base class: {pipeline.__class__.__bases__}")

# Test 1: Text-to-video mode (generates 16 frames)
print("\n=== Test 1: Text-to-Video Mode ===")
prompt = [{"text": "A cat walking through a forest, cinematic lighting", "weight": 100}]

start = time.time()
output_text = pipeline(prompts=prompt)
latency_text = time.time() - start

num_frames_text, h, w, c = output_text.shape
fps_text = num_frames_text / latency_text

print(
    f"test.py: Generated {num_frames_text} frames in {latency_text:.2f}s ({fps_text:.2f} fps)"
)
print(f"test.py: Output shape: {output_text.shape}")

# Save text mode output
output_text_np = output_text.detach().cpu().contiguous().numpy()
export_to_video(output_text_np, Path(__file__).parent / "output_text.mp4", fps=16)
print("test.py: Saved text-to-video output to output_text.mp4")

# Test 2: Video-to-video mode (uses first 4 frames as input)
print("\n=== Test 2: Video-to-Video Mode ===")
# Video input needs to be a list of individual frame tensors
# Each frame should be shape [1, H, W, C] for preprocess_chunk
# We use the first 4 frames from text mode output
video_input = [output_text[i : i + 1] for i in range(4)]  # Split into 4 separate frames
print(f"test.py: Video input format: list of {len(video_input)} tensors")
print(f"test.py: First tensor shape: {video_input[0].shape}")

start = time.time()
output_video = pipeline(video=video_input, prompts=prompt, generation_mode="video")
latency_video = time.time() - start

num_frames_video = output_video.shape[0]
fps_video = num_frames_video / latency_video

print(
    f"test.py: Generated {num_frames_video} frames in {latency_video:.2f}s ({fps_video:.2f} fps)"
)
print(f"test.py: Output shape: {output_video.shape}")

# Save video mode output
output_video_np = output_video.detach().cpu().contiguous().numpy()
export_to_video(output_video_np, Path(__file__).parent / "output_video.mp4", fps=16)
print("test.py: Saved video-to-video output to output_video.mp4")

# Summary
print("\n=== Test Summary ===")
print(f"Text mode: {num_frames_text} frames, {latency_text:.2f}s, {fps_text:.2f} fps")
print(
    f"Video mode: {num_frames_video} frames, {latency_video:.2f}s, {fps_video:.2f} fps"
)
print("\nMultiModePipeline architecture working correctly!")
