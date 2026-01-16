import math
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, PipelineOutput, Requirements
from ..process import postprocess_chunk, preprocess_chunk
from .schema import PassthroughConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig


class PassthroughPipeline(Pipeline):
    """Passthrough pipeline for testing"""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return PassthroughConfig

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.height = height
        self.width = width
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self.prompts = None

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor | PipelineOutput:
        input = kwargs.get("video")

        if input is None:
            raise ValueError("Input cannot be None for PassthroughPipeline")

        if isinstance(input, list):
            # Don't resize for passthrough - preserve original input resolution
            input = preprocess_chunk(input, self.device, self.dtype)

        input = rearrange(input, "B C T H W -> B T C H W")

        video_tensor = postprocess_chunk(input)

        # Generate beep audio matching video duration
        # Default FPS: 16 (common for video generation)
        # Audio sample rate: 24000 Hz (matching LTX2 pipeline)
        fps = kwargs.get("fps", 16.0)
        audio_sample_rate = 24000

        # Get number of frames from video tensor (THWC format)
        num_frames = video_tensor.shape[0]
        video_duration_seconds = num_frames / fps

        # Calculate number of audio samples needed
        num_audio_samples = int(audio_sample_rate * video_duration_seconds)

        # Generate beep tones: alternating frequencies for variety
        # Use multiple beeps at different frequencies
        beep_frequency_1 = 440.0  # A4 note
        beep_frequency_2 = 523.25  # C5 note

        # Create time array
        t = torch.linspace(0, video_duration_seconds, num_audio_samples, device=self.device, dtype=torch.float32)

        # Generate beeps: alternate between two frequencies every 0.5 seconds
        beep_period = 0.5  # seconds
        # Determine which frequency to use for each sample (alternating every 0.5s)
        period_index = (t / beep_period).long()
        freq_mask = (period_index % 2 == 0)
        frequencies = torch.where(freq_mask, beep_frequency_1, beep_frequency_2)

        # Generate envelope to avoid clicks (triangular envelope within each period)
        period_position = t % beep_period
        envelope = 0.3 * (1.0 - torch.abs(period_position - beep_period / 2) / (beep_period / 2))

        # Generate sine wave with varying frequency and envelope
        beep_wave = envelope * torch.sin(2 * math.pi * frequencies * t)

        # Convert to stereo (2 channels)
        audio_tensor = torch.stack([beep_wave, beep_wave], dim=0)  # Shape: (2, num_samples)

        return PipelineOutput(
            video=video_tensor,
            audio=audio_tensor,
            audio_sample_rate=audio_sample_rate,
        )
