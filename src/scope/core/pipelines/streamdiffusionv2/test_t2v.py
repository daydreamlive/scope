import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import StreamDiffusionV2Pipeline

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
        "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
        "height": 480,
        "width": 832,
    }
)

device = torch.device("cuda")
pipeline = StreamDiffusionV2Pipeline(
    config,
    device=device,
    dtype=torch.bfloat16,
)

prompts = [{"text": "A panda in the grass, looking around", "weight": 100}]

outputs = []
latency_measures = []
fps_measures = []

# Generate ~1 second of video (16 frames at 16fps output)
# StreamDiffusionV2 generates 4 frames per call (num_frame_per_block=1 * vae_temporal_downsample_factor=4)
max_output_frames = 16
num_frames = 0

print(f"test_t2v: Generating up to {max_output_frames} frames")

while num_frames < max_output_frames:
    start = time.time()

    output = pipeline(prompts=prompts)

    num_output_frames, _, _, _ = output.shape
    latency = time.time() - start
    fps = num_output_frames / latency

    print(
        f"test_t2v: Pipeline generated {num_output_frames} frames latency={latency:.2f}s fps={fps:.2f}"
    )

    latency_measures.append(latency)
    fps_measures.append(fps)
    num_frames += num_output_frames
    outputs.append(output.detach().cpu())

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(f"test_t2v: Final output shape={output_video.shape}")
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, Path(__file__).parent / "output_t2v.mp4", fps=16)

# Print statistics
print("\n=== Performance Statistics ===")
print(f"Total chunks processed: {len(latency_measures)}")
print(f"Total frames generated: {num_frames}")
print(
    f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
    f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
)
print(
    f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
    f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
)
print(f"\nOutput saved to: {Path(__file__).parent / 'output_t2v.mp4'}")
