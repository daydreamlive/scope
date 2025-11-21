import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from ..video import load_video
from .pipeline import StreamDiffusionV2Pipeline

chunk_size = 4
start_chunk_size = 5

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

# input_video is a 1CTHW tensor
input_video = (
    load_video(
        Path(__file__).parent / "assets" / "original.mp4",
        resize_hw=(config.height, config.width),
    )
    .unsqueeze(0)
    .to("cuda", torch.bfloat16)
)
_, _, num_frames, _, _ = input_video.shape

num_chunks = (num_frames - 1) // chunk_size

prompts = [{"text": "a bear is walking on the grass", "weight": 100}]

outputs = []
latency_measures = []
fps_measures = []
start_idx = 0
end_idx = start_chunk_size
for i in range(num_chunks):
    if i > 0:
        start_idx = end_idx
        end_idx = end_idx + chunk_size

    chunk = input_video[:, :, start_idx:end_idx]

    start = time.time()
    # output is TCHW
    output = pipeline(video=chunk, prompts=prompts)

    num_output_frames, _, _, _ = output.shape
    latency = time.time() - start
    fps = num_output_frames / latency

    print(
        f"Pipeline generated {num_output_frames} frames latency={latency:2f}s fps={fps}"
    )

    latency_measures.append(latency)
    fps_measures.append(fps)
    outputs.append(output.detach().cpu())

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(output_video.shape)
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, Path(__file__).parent / "output.mp4", fps=16)

# Print statistics
print("\n=== Performance Statistics ===")
print(
    f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
)
print(
    f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
)
