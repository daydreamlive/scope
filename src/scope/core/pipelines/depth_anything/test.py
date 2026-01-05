import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from ..video import load_video
from .pipeline import DepthAnythingPipeline

config = OmegaConf.create(
    {
        "metric": False,
        "input_size": 518,
        "fp32": False,
        "height": 480,
        "width": 832,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = DepthAnythingPipeline(config, device=device)

# Load video and convert to list of THWC tensors
video_tensor = load_video(
    Path(__file__).parent / "assets" / "original.mp4",
    resize_hw=(config.height, config.width),
    normalize=False,
)
# Convert from CTHW to list of THWC tensors [1, H, W, C]
num_frames = video_tensor.shape[1]
video_list = [
    video_tensor[:, i].permute(1, 2, 0).unsqueeze(0)  # CTHW -> [1, H, W, C]
    for i in range(num_frames)
]

start = time.time()
depths = pipeline(video=video_list)
latency = time.time() - start

num_frames_processed = depths.shape[0]
fps = num_frames_processed / latency

print(
    f"Pipeline processed {num_frames_processed} frames latency={latency:.2f}s fps={fps:.2f}"
)
print(f"Output shape: {depths.shape}")

# Save depth video
depths_np = depths.cpu().numpy()
# Normalize to [0, 1] for export_to_video
depths_np = (depths_np - depths_np.min()) / (depths_np.max() - depths_np.min() + 1e-8)
export_to_video(depths_np, Path(__file__).parent / "output.mp4", fps=30)
