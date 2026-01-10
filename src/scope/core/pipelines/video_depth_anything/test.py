import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.pipelines.utils import print_statistics

from ..video import load_video
from .pipeline import VideoDepthAnythingPipeline


def main():
    """Test the Video Depth Anything pipeline."""
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
    pipeline = VideoDepthAnythingPipeline(config, device=device)

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

    # Process frames one by one
    latency_measures = []
    fps_measures = []
    depths_list = []

    for i, frame in enumerate(video_list):
        start = time.time()
        # Call pipeline with single frame
        depth = pipeline(video=[frame])
        latency = time.time() - start

        num_output_frames, _, _, _ = depth.shape
        fps = num_output_frames / latency

        print(
            f"Pipeline processed frame {i + 1}/{num_frames} latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        depths_list.append(depth.cpu())

    # Concatenate all depth frames
    depths = torch.concat(depths_list)
    print(f"Output shape: {depths.shape}")

    # Save depth video
    output_path = Path(__file__).parent / "output.mp4"
    depths_np = depths.cpu().numpy()
    # Normalize to [0, 1] for export_to_video
    depths_np = (depths_np - depths_np.min()) / (
        depths_np.max() - depths_np.min() + 1e-8
    )
    export_to_video(depths_np, output_path, fps=30)

    print(f"Saved video to {output_path}")

    # Print statistics
    print_statistics(latency_measures, fps_measures)


if __name__ == "__main__":
    main()
