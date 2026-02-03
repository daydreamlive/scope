import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from scope.core.pipelines.utils import print_statistics

from ..video import load_video
from .pipeline import ScribblePipeline


def main():
    """Test the Scribble preprocessor pipeline."""
    height = 512
    width = 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = ScribblePipeline(device=device)

    # Load video from frontend assets (default test video)
    video_path = (
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "frontend"
        / "public"
        / "assets"
        / "test.mp4"
    )
    video_tensor = load_video(
        video_path,
        resize_hw=(height, width),
        normalize=False,
    )
    # Convert from CTHW to list of THWC tensors [1, H, W, C]
    num_frames = video_tensor.shape[1]
    video_list = [
        video_tensor[:, i].permute(1, 2, 0).unsqueeze(0)  # CTHW -> [1, H, W, C]
        for i in range(num_frames)
    ]

    print(f"Input video: {num_frames} frames at {height}x{width}")

    # Process all frames at once
    start = time.time()
    output_dict = pipeline(video=video_list)
    output = output_dict["video"]
    latency = time.time() - start

    num_output_frames = output.shape[0]
    fps = num_output_frames / latency

    print(f"Output shape: {output.shape}")
    print(f"Processing latency: {latency:.2f}s, FPS: {fps:.2f}")

    # Save scribble video
    output_path = Path(__file__).parent / "output.mp4"
    output_np = output.cpu().numpy()
    export_to_video(output_np, output_path, fps=30)

    print(f"Saved scribble video to {output_path}")

    # Print statistics
    print_statistics([latency], [fps])


if __name__ == "__main__":
    main()
