import os
import time

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..video import load_video


class VodPipeline(Pipeline):
    """VOD pipeline for video on demand processing"""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        output_fps: int = 30,
        video_path: str = os.path.join(os.path.dirname(__file__), "vod.mp4"),
    ):
        self.height = height
        self.width = width
        self.device = device if device is not None else torch.device("cuda")
        self.dtype = dtype
        self.output_fps = output_fps
        self.prompts = None
        self.video_path = video_path
        self.video_frames = None
        self.current_frame_index = 0
        self._load_video()

    def _load_video(self):
        """Load video frames from the specified video file."""
        if not os.path.exists(self.video_path):
            print(
                f"Error: Video file {self.video_path} not found. Using gray frames as fallback."
            )
            return

        try:
            # Load video as tensor with shape [C, T, H, W]
            video_tensor = load_video(
                self.video_path,
                resize_hw=(self.height, self.width),
                normalize=False,  # We'll handle normalization in the call method
            )

            # Convert to [T, C, H, W] format for easier indexing
            self.video_frames = rearrange(video_tensor, "C T H W -> T C H W")
            print(f"Loaded {self.video_frames.shape[0]} frames from {self.video_path}")
        except Exception as e:
            print(
                f"Error loading video {self.video_path}: {e}. Using gray frames as fallback."
            )

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        return None

    def __call__(
        self, input: torch.Tensor | list[torch.Tensor] | None = None, **kwargs
    ) -> torch.Tensor:
        # Simulate the inference delay
        time.sleep(1 / self.output_fps)

        # Get the next frame from the video
        frame_idx = self.current_frame_index % self.video_frames.shape[0]
        frame = self.video_frames[frame_idx]  # [C, H, W]

        # Normalize to [-1, 1] range
        frame = frame / 255.0 * 2.0 - 1.0

        # Move to device and convert dtype
        frame = frame.to(device=self.device, dtype=self.dtype)

        # Add time dimension: [C, H, W] -> [1, C, H, W]
        video_frames = frame.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

        self.current_frame_index += 1

        # Rearrange to BTCWH format as expected by postprocess_chunk
        chunk = rearrange(video_frames, "B T C H W -> B T C H W")

        return postprocess_chunk(chunk)
