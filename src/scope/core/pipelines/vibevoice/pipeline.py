"""VibeVoice Text-to-Speech Pipeline.

This pipeline generates speech audio from text input using the VibeVoice model.
For now, it returns audio chunks from a hardcoded output.wav file.
"""

import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import torch

from ..interface import Pipeline

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Audio constants matching VibeVoice
SAMPLE_RATE = 24000  # 24kHz sample rate
CHUNK_SIZE = 1920  # ~80ms chunks at 24kHz (1920 samples = 80ms)


class VibeVoicePipeline(Pipeline):
    """VibeVoice Text-to-Speech Pipeline.

    This pipeline takes text input and produces audio output in streaming chunks.
    Unlike video pipelines that return tensors in THWC format, this pipeline
    yields audio chunks as 1D tensors of float32 samples.

    The audio is returned at 24kHz sample rate.
    """

    # Class-level flag to indicate this is an audio pipeline
    is_audio_pipeline = True
    sample_rate = SAMPLE_RATE

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        from ..schema import VibeVoiceConfig

        return VibeVoiceConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self._current_text = ""
        self._audio_generator = None

        # For now, use hardcoded output.wav path
        # In production, this would be replaced with actual model inference
        self._output_wav_path = Path(__file__).parent / "output.wav"

        logger.info(f"VibeVoicePipeline initialized on device {self.device}")

    def prepare(self, **kwargs):
        """Prepare the pipeline for generation.

        For audio pipelines, we don't need video input requirements.
        """
        return None  # No video input requirements

    def _load_wav_as_chunks(self) -> Generator[np.ndarray, None, None]:
        """Load the hardcoded output.wav file and yield it as chunks.

        Returns:
            Generator yielding audio chunks as numpy arrays.
        """
        if not self._output_wav_path.exists():
            logger.warning(
                f"output.wav not found at {self._output_wav_path}, generating silence"
            )
            # Generate 2 seconds of silence as a fallback
            total_samples = SAMPLE_RATE * 2
            for i in range(0, total_samples, CHUNK_SIZE):
                chunk_samples = min(CHUNK_SIZE, total_samples - i)
                yield np.zeros(chunk_samples, dtype=np.float32)
            return

        try:
            with wave.open(str(self._output_wav_path), "rb") as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()

                logger.info(
                    f"Loading WAV: {n_channels} channels, {sample_width} bytes/sample, "
                    f"{framerate}Hz, {n_frames} frames"
                )

                # Read all frames
                audio_data = wav_file.readframes(n_frames)

                # Convert bytes to numpy array based on sample width
                if sample_width == 2:
                    # 16-bit PCM
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    # Normalize to [-1.0, 1.0]
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif sample_width == 4:
                    # 32-bit PCM
                    audio_array = np.frombuffer(audio_data, dtype=np.int32)
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    # Assume 8-bit unsigned
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                    audio_array = (audio_array.astype(np.float32) - 128.0) / 128.0

                # If stereo, convert to mono by averaging channels
                if n_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1)

                # Resample if necessary (simple nearest-neighbor)
                if framerate != SAMPLE_RATE:
                    logger.info(f"Resampling from {framerate}Hz to {SAMPLE_RATE}Hz")
                    # Calculate new length
                    new_length = int(len(audio_array) * SAMPLE_RATE / framerate)
                    # Resample using linear interpolation
                    old_indices = np.linspace(0, len(audio_array) - 1, new_length)
                    audio_array = np.interp(
                        old_indices, np.arange(len(audio_array)), audio_array
                    )

                # Yield chunks
                for i in range(0, len(audio_array), CHUNK_SIZE):
                    chunk = audio_array[i : i + CHUNK_SIZE]
                    yield chunk.astype(np.float32)

        except Exception as e:
            logger.error(f"Error loading WAV file: {e}")
            # Yield silence on error
            yield np.zeros(CHUNK_SIZE, dtype=np.float32)

    def generate_audio(self, text: str) -> Generator[torch.Tensor, None, None]:
        """Generate audio from text.

        This is a generator that yields audio chunks as they are generated.
        For now, it yields chunks from the hardcoded output.wav file.

        Args:
            text: The input text to synthesize.

        Yields:
            Audio chunks as 1D torch tensors of float32 samples.
        """
        logger.info(f"Generating audio for text: {text[:50]}...")

        # For now, just load and yield chunks from output.wav
        for chunk in self._load_wav_as_chunks():
            # Convert to torch tensor
            chunk_tensor = torch.from_numpy(chunk).to(dtype=self.dtype)
            yield chunk_tensor

    def __call__(
        self,
        text: str | None = None,
        prompts: list | None = None,
        **kwargs,
    ) -> Generator[torch.Tensor, None, None]:
        """Generate audio from text input.

        This method is a generator that yields audio chunks.

        Args:
            text: The text to synthesize into speech.
            prompts: Alternative way to pass text (uses first prompt's text).
            **kwargs: Additional parameters (ignored for now).

        Yields:
            Audio chunks as 1D torch tensors of float32 samples.
        """
        # Extract text from prompts if text not provided directly
        if text is None and prompts:
            if isinstance(prompts[0], dict):
                text = prompts[0].get("text", "")
            elif isinstance(prompts[0], str):
                text = prompts[0]
            else:
                text = str(prompts[0])

        if not text:
            text = "Hello, this is a test of the VibeVoice text-to-speech system."
            logger.warning(f"No text provided, using default: {text}")

        self._current_text = text

        # Return the generator
        return self.generate_audio(text)

    def get_next_chunk(self) -> torch.Tensor | None:
        """Get the next audio chunk from the current generation.

        This is used by the audio processor to get chunks one at a time.

        Returns:
            The next audio chunk, or None if generation is complete.
        """
        if self._audio_generator is None:
            return None

        try:
            return next(self._audio_generator)
        except StopIteration:
            self._audio_generator = None
            return None

    def start_generation(self, text: str):
        """Start a new audio generation.

        Args:
            text: The text to synthesize.
        """
        self._current_text = text
        self._audio_generator = self.generate_audio(text)

    def is_generating(self) -> bool:
        """Check if audio generation is in progress.

        Returns:
            True if generation is in progress, False otherwise.
        """
        return self._audio_generator is not None
