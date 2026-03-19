import math
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline
from .schema import AudioBeepConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

# Generate audio at 48 kHz to match WebRTC output rate (avoids resampling).
SAMPLE_RATE = 48000

# Hard cap: never generate more than 1 second of audio in a single call.
# Prevents a large burst after long pauses or parameter changes.
MAX_SAMPLES_PER_CALL = SAMPLE_RATE

# Minimum chunk size before producing output. Matches WebRTC 20ms frame cadence
# and prevents flooding the output queue with many tiny chunks.
MIN_CHUNK_SAMPLES = int(0.02 * SAMPLE_RATE)  # 960 samples = 20ms

# Short fade at beep edges to avoid clicks from abrupt signal discontinuities.
FADE_DURATION = 0.002  # 2ms = 96 samples at 48kHz


class AudioBeepPipeline(Pipeline):
    """Audio-only pipeline that generates periodic beep tones.

    Uses wall-clock tracking to produce the correct number of samples per call,
    compensating for framework sleep and processing overhead. A free-running
    oscillator with gated envelope avoids phase discontinuities and clicks.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return AudioBeepConfig

    def __init__(self, **kwargs):
        self._clock_start: float | None = None
        self._samples_produced = 0

    def __call__(self, **kwargs) -> dict:
        frequency = kwargs.get("frequency", 440.0)
        beep_duration = kwargs.get("beep_duration", 0.1)
        beep_interval = kwargs.get("beep_interval", 1.0)
        volume = kwargs.get("volume", 0.5)

        # Determine how many samples to generate based on wall-clock time.
        # One chunk (20ms) of lead keeps the consumer's buffer fed despite jitter.
        now = time.monotonic()
        if self._clock_start is None:
            self._clock_start = now
            self._samples_produced = 0

        elapsed = now - self._clock_start + 0.02
        target_samples = int(elapsed * SAMPLE_RATE)
        n_samples = target_samples - self._samples_produced
        n_samples = min(n_samples, MAX_SAMPLES_PER_CALL)

        if n_samples < MIN_CHUNK_SAMPLES:
            time.sleep(0.02)
            return {}

        # Free-running oscillator: global sample index keeps phase continuous
        # regardless of beep/silence gating. Modulo keeps the argument small
        # to preserve float precision over long runs.
        global_indices = self._samples_produced + torch.arange(
            n_samples, dtype=torch.float64
        )
        phase = (2.0 * math.pi * frequency / SAMPLE_RATE * global_indices) % (
            2.0 * math.pi
        )
        sine = torch.sin(phase.float())

        # Position within the beep cycle for each sample
        global_time = global_indices / SAMPLE_RATE
        pos_in_cycle = global_time % beep_interval

        # Smooth envelope: fade in at beep start, fade out at beep end
        fade = min(FADE_DURATION, beep_duration / 2)
        envelope = torch.zeros(n_samples, dtype=torch.float32)

        in_beep = pos_in_cycle < beep_duration
        if in_beep.any():
            pos_beep = pos_in_cycle[in_beep]

            env_values = torch.ones(pos_beep.shape[0], dtype=torch.float32)

            fade_in = pos_beep < fade
            if fade_in.any():
                env_values[fade_in] = (pos_beep[fade_in] / fade).float()

            fade_out = pos_beep >= beep_duration - fade
            if fade_out.any():
                env_values[fade_out] = (
                    (beep_duration - pos_beep[fade_out]) / fade
                ).float()

            envelope[in_beep] = env_values

        audio = volume * envelope * sine

        self._samples_produced += n_samples

        # Stereo: duplicate mono to both channels — shape (2, N)
        stereo = torch.stack([audio, audio])

        return {
            "audio": stereo,
            "audio_sample_rate": SAMPLE_RATE,
        }
