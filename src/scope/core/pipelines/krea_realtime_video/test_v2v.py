import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import v2

from scope.core.config import get_model_file_path, get_models_dir

from ..utils import Quantization
from .pipeline import KreaRealtimeVideoPipeline


def load_video_cv2(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads a video as a CTHW tensor using OpenCV (Windows-compatible).
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        if num_frames is not None and frame_count >= num_frames:
            break

    cap.release()

    # Stack frames: list of HWC -> THWC
    video = torch.from_numpy(np.stack(frames))
    # Rearrange to TCHW for resize
    video = rearrange(video, "T H W C -> T C H W")

    height, width = video.shape[2:]
    if resize_hw is not None and (height != resize_hw[0] or width != resize_hw[1]):
        video = v2.Resize(resize_hw, antialias=True)(video)

    video = video.float()

    if normalize:
        # Normalize to [-1, 1]
        video = video / 127.5 - 1.0

    # Rearrange to CTHW
    video = rearrange(video, "T C H W -> C T H W")

    return video


# Krea uses num_frame_per_block=3 and vae_temporal_downsample_factor=4
# So video input size is 3 * 4 = 12 frames per chunk
chunk_size = 12
start_chunk_size = 12

config = OmegaConf.create(
    {
        "model_dir": str(get_models_dir()),
        "generator_path": str(
            get_model_file_path(
                "krea-realtime-video/krea-realtime-video-14b.safetensors"
            )
        ),
        "text_encoder_path": str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        ),
        "tokenizer_path": str(get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")),
        "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
        "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
        # Small resolution for VRAM constraints
        "height": 256,
        "width": 256,
    }
)

device = torch.device("cuda")
pipeline = KreaRealtimeVideoPipeline(
    config,
    # Use FP8 quantization for VRAM savings
    quantization=Quantization.FP8_E4M3FN,
    compile=False,
    device=device,
    dtype=torch.bfloat16,
)

# Load the default test video from the frontend assets
default_video_path = (
    Path(__file__).parent.parent.parent.parent.parent.parent
    / "frontend"
    / "public"
    / "assets"
    / "test.mp4"
)
print(f"test_v2v: Loading video from {default_video_path}")

# Load only ~2 second of video (24 frames at typical video framerates)
# This gives us 4 chunks of 12 frames each
max_frames = 48

input_video = (
    load_video_cv2(
        default_video_path,
        num_frames=max_frames,
        resize_hw=(config.height, config.width),
    )
    .unsqueeze(0)
    .to("cuda", torch.bfloat16)
)
_, _, num_frames, _, _ = input_video.shape
print(f"test_v2v: Loaded video with {num_frames} frames, shape={input_video.shape}")

num_chunks = (num_frames - start_chunk_size) // chunk_size + 1
print(f"test_v2v: Processing {num_chunks} chunks (chunk_size={chunk_size})")

prompts = [
    {
        "text": "A 3D animated scene. A **panda** sitting in the grass, looking around.",
        "weight": 100,
    }
]

outputs = []
latency_measures = []
fps_measures = []
start_idx = 0
end_idx = start_chunk_size

for i in range(num_chunks):
    if i > 0:
        start_idx = end_idx
        end_idx = end_idx + chunk_size

    # Clamp end_idx to not exceed video length
    end_idx = min(end_idx, num_frames)

    chunk = input_video[:, :, start_idx:end_idx]
    chunk_frames = chunk.shape[2]

    print(
        f"test_v2v: Processing chunk {i + 1}/{num_chunks} (frames {start_idx}-{end_idx}, size={chunk_frames})"
    )

    start = time.time()
    # output is THWC
    output = pipeline(
        video=chunk,
        prompts=prompts,
        kv_cache_attention_bias=0.3,
        denoising_step_list=[1000, 750],
        noise_scale=0.7,
    )

    num_output_frames, _, _, _ = output.shape
    latency = time.time() - start
    fps = num_output_frames / latency

    print(
        f"test_v2v: Pipeline generated {num_output_frames} frames latency={latency:.2f}s fps={fps:.2f}"
    )

    latency_measures.append(latency)
    fps_measures.append(fps)
    outputs.append(output.detach().cpu())

    # Stop if we've reached the end of the video
    if end_idx >= num_frames:
        break

# Concatenate all of the THWC tensors
output_video = torch.concat(outputs)
print(f"test_v2v: Final output shape={output_video.shape}")
output_video_np = output_video.contiguous().numpy()
export_to_video(output_video_np, Path(__file__).parent / "output_v2v.mp4", fps=16)

# Print statistics
print("\n=== Performance Statistics ===")
print(f"Total chunks processed: {len(latency_measures)}")
print(f"Total frames processed: {sum(o.shape[0] for o in outputs)}")
print(
    f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
    f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
)
print(
    f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
    f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
)
print(f"\nOutput saved to: {Path(__file__).parent / 'output_v2v.mp4'}")
