"""
Modular LongLive + ControlNet canny spike.

This script mirrors the previous research branch behavior using:
- The Wan ControlNet teacher from notes/wan2.1-dilated-controlnet
- Real pretrained canny weights from HuggingFace
- Canny edges extracted from a reference video as control frames

Usage (example):
    uv run -m pipelines.longlive.test_controlnet_modular --num-chunks 2
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir

from .extract_canny_edges import extract_canny_edges
from .pipeline import LongLivePipeline


def _load_control_frames(
    video_path: str,
    num_teacher_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Load canny edges and convert to ControlNet frames tensor."""
    edges_np = extract_canny_edges(video_path, num_teacher_frames, height, width)
    if edges_np is None:
        # Fallback to white frames if extraction failed
        edges_np = np.ones((num_teacher_frames, height, width, 3), dtype=np.float32)

    # Normalize to [0, 1] and convert to torch [B, C, T, H, W]
    edges_normalized = edges_np / 255.0
    edges_torch = (
        torch.from_numpy(edges_normalized)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device)
        .to(torch.bfloat16)
    )
    return edges_torch


def main():
    parser = argparse.ArgumentParser(
        description="Test modular LongLive + ControlNet Canny conditioning"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="Number of chunks to generate (default: 2)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MODULAR LONGLIVE + CONTROLNET CANNY SPIKE")
    print("=" * 80)

    device = torch.device("cuda")
    height = 480
    width = 832
    num_chunks = args.num_chunks

    # Load LongLive config
    config_path = Path("pipelines/longlive/model.yaml")
    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(str(config_path)),
            "height": height,
            "width": width,
        }
    )

    print("\nLoading LongLive pipeline...")
    pipeline = LongLivePipeline(
        config,
        device=device,
        dtype=torch.bfloat16,
    )

    # Prepare ControlNet control frames buffer (Canny edges)
    print("\nPreparing Canny control frames buffer...")
    video_path = "pipelines/streamdiffusionv2/assets/original.mp4"

    # 3 frames per chunk (same as LongLive num_frame_per_block),
    # teacher uses 4x temporal compression.
    frames_per_chunk = 3
    total_student_frames = num_chunks * frames_per_chunk
    compression_ratio = 4
    total_teacher_frames = total_student_frames * compression_ratio

    control_frames_buffer = _load_control_frames(
        video_path, total_teacher_frames, height, width, device
    )

    pipeline.state.set("control_frames_buffer", control_frames_buffer)
    pipeline.state.set("controlnet_weight", 1.0)
    pipeline.state.set("controlnet_stride", 3)
    pipeline.state.set("controlnet_compression_ratio", compression_ratio)

    # Simple test prompt
    prompt = "A cat sitting in the grass looking back and forth"

    # First call initializes caches
    print("\nPreparing text condition and caches...")
    pipeline(prompts=prompt)

    print(f"\nGenerating {num_chunks} chunks with ControlNet Canny conditioning...")
    outputs: list[torch.Tensor] = []

    for chunk_idx in range(num_chunks):
        print(f"\nGenerating chunk {chunk_idx + 1}/{num_chunks}...")
        start = time.time()
        output = pipeline(prompts=prompt)
        elapsed = time.time() - start
        print(
            f"Chunk {chunk_idx + 1} generated {output.shape[0]} frames in {elapsed:.2f}s"
        )
        outputs.append(output.detach().cpu())

    if not outputs:
        print("\nNo output generated - all chunks failed")
        return

    full_output = torch.cat(outputs, dim=0)
    print(f"\nTotal generated frames: {full_output.shape[0]}")

    output_np = full_output.numpy()
    output_path = "pipelines/longlive/output_controlnet_modular_canny.mp4"
    export_to_video(output_np, output_path, fps=16)
    print(f"\nSaved video to {output_path}")
    print("Compare visually against pipelines/longlive/control_frames_canny.mp4")


if __name__ == "__main__":
    main()
