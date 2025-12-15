"""
Generate an all-white mask video for testing inpainting.

Creates a video file with all white frames (1.0, 1.0, 1.0) which means
everything should be inpainted (no preserved regions).

Uses the same methodology as test_inpainting.py (export_to_video from diffusers).

Usage:
    python -m scope.core.pipelines.longlive.generate_white_mask
"""

from pathlib import Path

import numpy as np
from diffusers.utils import export_to_video


def generate_white_mask_video(
    output_path: str | Path,
    width: int = 512,
    height: int = 512,
    num_frames: int = 48,
    fps: int = 16,
):
    """
    Generate an all-white mask video.

    Args:
        output_path: Path to save the output video file
        width: Video width in pixels
        height: Video height in pixels
        num_frames: Number of frames in the video
        fps: Frames per second
    """
    print(f"generate_white_mask_video: Creating {num_frames} frame white mask video")
    print(f"generate_white_mask_video: Resolution: {width}x{height}, FPS: {fps}")
    print(f"generate_white_mask_video: Output: {output_path}")

    # Create all-white frames as numpy array [F, H, W, C] with values in [0, 1]
    # White = 1.0 means everything will be inpainted (no preserved regions)
    white_frames = np.ones((num_frames, height, width, 3), dtype=np.float32)

    # Export using same method as test_inpainting.py
    export_to_video(white_frames, str(output_path), fps=fps)

    print(
        f"generate_white_mask_video: Successfully created white mask video: {output_path}"
    )


def main():
    """Generate white mask video."""
    # Determine output path (same directory as this script)
    script_dir = Path(__file__).parent
    output_path = script_dir / "white_mask_512x512.mp4"

    # Generate video
    generate_white_mask_video(
        output_path=output_path,
        width=512,
        height=512,
        num_frames=48,  # 4 chunks of 12 frames each
        fps=16,  # Match test_inpainting.py fps
    )

    print(f"\nWhite mask video saved to: {output_path}")
    print("This mask means everything will be inpainted (no preserved regions).")


if __name__ == "__main__":
    main()
