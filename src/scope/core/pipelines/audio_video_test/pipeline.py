import math
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline
from .schema import AudioVideoTestConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

# Generate audio at 48 kHz to match WebRTC output rate (avoids resampling).
SAMPLE_RATE = 48000

# Hard cap: never generate more than 1 second of audio in a single call.
MAX_SAMPLES_PER_CALL = SAMPLE_RATE

# How far ahead of wall-clock to pre-generate audio (seconds).
# A larger buffer means audio is produced in fewer, bigger batches so most
# video-frame calls skip audio entirely, reducing per-frame overhead.
AUDIO_LOOKAHEAD = 0.200  # 200ms — keeps ~5 out of 6 video calls audio-free at 30fps

# Minimum audio chunk size before producing output. Matches WebRTC 20ms frame
# cadence and prevents flooding the audio output queue with many tiny chunks
# when the video generation loop runs much faster than the audio consumer.
MIN_CHUNK_SAMPLES = int(0.02 * SAMPLE_RATE)  # 960 samples = 20ms

# Short fade at beep edges to avoid clicks from abrupt signal discontinuities.
FADE_DURATION = 0.002  # 2ms = 96 samples at 48kHz


class AudioVideoTestPipeline(Pipeline):
    """Generates periodic beep tones with a flashing video frame.

    Video: white frame during beep, black during silence.
    Audio: sine wave beep with smooth fade envelope.

    Uses wall-clock tracking for audio pacing. Video is produced every call
    (the framework paces video via its own frame loop).
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioVideoTestConfig

    def __init__(self, height: int = 512, width: int = 512, **kwargs):
        self.height = height
        self.width = width
        self._clock_start: float | None = None
        self._samples_produced = 0

    def __call__(self, **kwargs) -> dict:
        frequency = kwargs.get("frequency", 440.0)
        beep_duration = kwargs.get("beep_duration", 0.1)
        beep_interval = kwargs.get("beep_interval", 1.0)
        volume = kwargs.get("volume", 0.5)

        now = time.monotonic()
        if self._clock_start is None:
            self._clock_start = now
            self._samples_produced = 0

        elapsed = now - self._clock_start

        # --- Video: flash white during beep, black during silence ---
        pos_in_cycle = elapsed % beep_interval
        in_beep = pos_in_cycle < beep_duration
        brightness = 1.0 if in_beep else 0.0
        frame = torch.full(
            (1, self.height, self.width, 3), brightness, dtype=torch.float32
        )

        # --- Audio: wall-clock paced beep tone ---
        target_samples = int((elapsed + AUDIO_LOOKAHEAD) * SAMPLE_RATE)
        n_samples = target_samples - self._samples_produced
        n_samples = min(n_samples, MAX_SAMPLES_PER_CALL)

        result = {"video": frame}

        if n_samples < MIN_CHUNK_SAMPLES:
            return result

        # Free-running oscillator with modulo to preserve float precision
        global_indices = self._samples_produced + torch.arange(
            n_samples, dtype=torch.float64
        )
        phase = (2.0 * math.pi * frequency / SAMPLE_RATE * global_indices) % (
            2.0 * math.pi
        )
        sine = torch.sin(phase.float())

        # Position within the beep cycle for each audio sample
        global_time = global_indices / SAMPLE_RATE
        audio_pos_in_cycle = global_time % beep_interval

        # Smooth envelope: fade in at beep start, fade out at beep end
        fade = min(FADE_DURATION, beep_duration / 2)
        envelope = torch.zeros(n_samples, dtype=torch.float32)

        audio_in_beep = audio_pos_in_cycle < beep_duration
        if audio_in_beep.any():
            pos_beep = audio_pos_in_cycle[audio_in_beep]

            env_values = torch.ones(pos_beep.shape[0], dtype=torch.float32)

            fade_in = pos_beep < fade
            if fade_in.any():
                env_values[fade_in] = (pos_beep[fade_in] / fade).float()

            fade_out = pos_beep >= beep_duration - fade
            if fade_out.any():
                env_values[fade_out] = (
                    (beep_duration - pos_beep[fade_out]) / fade
                ).float()

            envelope[audio_in_beep] = env_values

        audio = volume * envelope * sine
        self._samples_produced += n_samples

        # Stereo: duplicate mono to both channels — shape (2, N)
        stereo = torch.stack([audio, audio])

        result["audio"] = stereo
        result["audio_sample_rate"] = SAMPLE_RATE

        return result
