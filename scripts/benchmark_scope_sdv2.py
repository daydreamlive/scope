"""Benchmark Scope's StreamDiffusionV2Pipeline (no plugins).

Direct comparison with the original repo benchmark.

Usage:
    uv run python scripts/benchmark_scope_sdv2.py
    uv run python scripts/benchmark_scope_sdv2.py --height 512 --width 512
    uv run python scripts/benchmark_scope_sdv2.py --height 208 --width 208
"""

import argparse
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
from scope.core.pipelines.video import load_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("StreamDiffusionV2/wan_causal_dmd_v2v/model.pt")
            ),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(
                Path(__file__).parent
                / "../src/scope/core/pipelines/streamdiffusionv2/model.yaml"
            ),
            "height": args.height,
            "width": args.width,
        }
    )

    device = torch.device("cuda")
    print("Loading Scope StreamDiffusionV2Pipeline...")
    t0 = time.perf_counter()
    pipeline = StreamDiffusionV2Pipeline(config, device=device, dtype=torch.bfloat16)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")

    # Load video
    video_path = "C:/_dev/StreamDiffusionV2/examples/original.mp4"
    input_video = (
        load_video(video_path, resize_hw=(args.height, args.width))
        .unsqueeze(0)
        .to("cuda", torch.bfloat16)
    )
    _, _, num_frames, _, _ = input_video.shape
    print(f"Video: {num_frames} frames at {args.width}x{args.height}")

    chunk_size = 4
    start_chunk_size = 5
    num_chunks = (num_frames - 1) // chunk_size

    prompts = [{"text": "A dog walks on the grass, realistic", "weight": 100}]

    latencies = []
    fps_values = []

    start_idx = 0
    end_idx = start_chunk_size

    print(f"\nBenchmarking {num_chunks} chunks, warmup={args.warmup}...\n")

    for i in range(num_chunks):
        if i > 0:
            start_idx = end_idx
            end_idx = end_idx + chunk_size

        if end_idx > num_frames:
            break

        chunk = input_video[:, :, start_idx:end_idx]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        output_dict = pipeline(video=chunk, prompts=prompts)
        output = output_dict["video"]

        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        n_out = output.shape[0]
        fps = n_out / latency

        if i >= args.warmup:
            latencies.append(latency)
            fps_values.append(fps)
            print(
                f"  Chunk {i:3d}: "
                f"{n_out} frames  "
                f"latency={latency * 1000:6.1f}ms  "
                f"fps={fps:.1f}"
            )

    if not fps_values:
        print("No measurements!")
        return

    def avg(xs):
        return sum(xs) / len(xs)

    print(f"\n{'=' * 60}")
    print(
        f"Scope StreamDiffusionV2 Results ({len(fps_values)} chunks at {args.width}x{args.height})"
    )
    print(f"{'=' * 60}")
    print(f"  Avg latency: {avg(latencies) * 1000:.1f}ms")
    print(f"  Avg FPS:     {avg(fps_values):.1f}")
    print(f"  Min FPS:     {min(fps_values):.1f}")
    print(f"  Max FPS:     {max(fps_values):.1f}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / (1024**3):.1f} GB")


if __name__ == "__main__":
    main()
