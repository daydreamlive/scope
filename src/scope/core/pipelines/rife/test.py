import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.pipelines.utils import print_statistics

from ..video import load_video
from .pipeline import RIFEPipeline


def main():
    """Test the RIFE frame interpolation pipeline."""
    config = OmegaConf.create(
        {
            "height": 480,
            "width": 832,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = RIFEPipeline(config, device=device)

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

    print(f"Input video: {num_frames} frames at {config.height}x{config.width}")

    # Process all frames at once
    start = time.time()
    output = pipeline(video=video_list)
    latency = time.time() - start

    num_output_frames = output.shape[0]
    fps = num_output_frames / latency

    print(f"Output shape: {output.shape}")
    print(f"Input frames: {num_frames}, Output frames: {num_output_frames}")
    print(f"Frame rate multiplier: {num_output_frames / num_frames:.1f}x")
    print(f"Processing latency: {latency:.2f}s, FPS: {fps:.2f}")

    # Verify output has higher frame count
    expected_frames = num_frames * 2 - 1
    if num_output_frames == expected_frames:
        print(
            f"SUCCESS: Frame count increased from {num_frames} to {num_output_frames}"
        )
    else:
        print(
            f"WARNING: Expected {expected_frames} frames, got {num_output_frames} frames"
        )

    # Save interpolated video
    output_path = Path(__file__).parent / "output.mp4"
    output_np = output.cpu().numpy()
    # Output is already in [0, 1] range from pipeline
    export_to_video(output_np, output_path, fps=60)  # Higher FPS for interpolated video

    print(f"Saved interpolated video to {output_path}")

    # Print statistics
    print_statistics([latency], [fps])


if __name__ == "__main__":
    main()
