import time

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir

from ..video import load_video
from .pipeline import StreamDiffusionV2Pipeline

config = OmegaConf.load("pipelines/streamdiffusionv2/model.yaml")

models_dir = get_models_dir()
height = 480
width = 832

chunk_size = 4
start_chunk_size = 5

config["model_dir"] = str(models_dir)
config["text_encoder_path"] = str(
    get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
)
config["height"] = height
config["width"] = width

pipeline = StreamDiffusionV2Pipeline(
    config,
    chunk_size=chunk_size,
    start_chunk_size=start_chunk_size,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
)
pipeline.prepare(prompts=["a bear is walking on the grass"])

# input_video is a 1CTHW tensor
input_video = (
    load_video(
        "pipelines/streamdiffusionv2/assets/original.mp4", resize_hw=(height, width)
    )
    .unsqueeze(0)
    .to("cuda", torch.bfloat16)
)
_, _, num_frames, _, _ = input_video.shape

num_chunks = (num_frames - 1) // chunk_size

outputs = []
start_idx = 0
end_idx = start_chunk_size
for i in range(num_chunks):
    if i > 0:
        start_idx = end_idx
        end_idx = end_idx + chunk_size

    chunk = input_video[:, :, start_idx:end_idx]

    start = time.time()
    # output is TCHW
    output = pipeline(chunk)

    num_output_frames, _, _, _ = output.shape
    latency = time.time() - start
    fps = num_output_frames / latency

    print(
        f"Pipeline generated {num_output_frames} frames latency={latency:2f}s fps={fps}"
    )

    outputs.append(output.detach().cpu())

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(output_video.shape)
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, "pipelines/streamdiffusionv2/output.mp4", fps=16)
