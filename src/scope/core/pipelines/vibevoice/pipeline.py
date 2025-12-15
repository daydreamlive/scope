import copy
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..interface import Pipeline, Requirements
from ..schema import VibeVoiceConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig


logger = logging.getLogger(__name__)


def _resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Naive linear resampling to match WebRTC's expected sample rate."""
    if source_rate == target_rate or audio.size == 0:
        return audio

    # Generate target indices and interpolate
    duration = audio.shape[0] / source_rate
    target_length = int(duration * target_rate)
    target_positions = np.linspace(0, audio.shape[0] - 1, num=target_length)
    resampled = np.interp(target_positions, np.arange(audio.shape[0]), audio)
    return resampled.astype(audio.dtype)


def _generate_fallback_tone(duration_seconds: float, sample_rate: int) -> np.ndarray:
    """Generate a short fallback tone so the player has audible output."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    # 440Hz sine wave scaled to int16
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    return (tone * 32767).astype(np.int16)


class VibeVoicePipeline(Pipeline):
    """Real-time text-to-speech pipeline using Microsoft VibeVoice."""

    # VibeVoice outputs at 24kHz, we upsample to 48kHz for WebRTC
    vibevoice_sample_rate = 24_000
    target_sample_rate = 48_000
    default_chunk_size = 960  # 20ms @ 48k, aligns with common WebRTC audio pacing

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VibeVoiceConfig

    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        speaker_name: str = "Emma",
        device: str | None = None,
        chunk_size: int | None = None,
        cfg_scale: float = 1.5,
    ):
        """Initialize VibeVoice pipeline.

        Args:
            model_path: Path to the HuggingFace model
            speaker_name: Name of the speaker voice to use
            device: Device for inference (cuda/mps/cpu), auto-detected if None
            chunk_size: Size of audio chunks to stream
            cfg_scale: Classifier-free guidance scale for generation
        """
        self.model_path = model_path
        self.speaker_name = speaker_name
        self.chunk_size = chunk_size or self.default_chunk_size
        self.cfg_scale = cfg_scale

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Normalize 'mpx' typo to 'mps'
        if device.lower() == "mpx":
            device = "mps"

        # Validate mps availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        self.device = device
        logger.info(f"VibeVoice using device: {self.device}")

        # Lazy initialization of model components
        self._model = None
        self._processor = None
        self._voice_mapper = None
        self._voice_sample = None

        # Audio buffer for streaming
        self._audio_buffer: np.ndarray | None = None
        self._position = 0
        self._lock = threading.Lock()
        self._last_text: str | None = None

    def _ensure_initialized(self):
        """Lazy load model and processor on first use."""
        if self._model is not None:
            return

        logger.info(f"Loading VibeVoice model from {self.model_path}")

        try:
            # Import VibeVoice modules
            from vibevoice.modular.modeling_vibevoice_streaming_inference import (
                VibeVoiceStreamingForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_streaming_processor import (
                VibeVoiceStreamingProcessor,
            )
        except ImportError as e:
            logger.error("Failed to import VibeVoice modules. Make sure VibeVoice is installed.")
            logger.error("Install from: /home/user/VibeVoice")
            raise RuntimeError(
                "VibeVoice not available. Please install it from ~/VibeVoice"
            ) from e

        # Load processor
        self._processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        # Determine dtype and attention implementation based on device
        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:  # cpu
            load_dtype = torch.float32
            attn_impl = "sdpa"

        logger.info(f"Loading model with dtype={load_dtype}, attn_implementation={attn_impl}")

        # Load model with device-specific settings
        try:
            if self.device == "mps":
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self._model.to("mps")
            elif self.device == "cuda":
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            else:  # cpu
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cpu",
                    attn_implementation=attn_impl,
                )
        except Exception as e:
            if attn_impl == "flash_attention_2":
                logger.warning(f"Failed with flash_attention_2: {e}")
                logger.warning("Retrying with SDPA (may result in lower audio quality)")
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation="sdpa",
                )
                if self.device == "mps":
                    self._model.to("mps")
            else:
                raise

        self._model.eval()
        self._model.set_ddpm_inference_steps(num_steps=5)

        # Load voice sample
        self._load_voice_sample()

        logger.info("VibeVoice model loaded successfully")

    def _load_voice_sample(self):
        """Load the voice sample for the specified speaker."""
        # Try to find voices directory in multiple locations:
        # 1. From VIBEVOICE_VOICES_DIR environment variable
        # 2. In a local VibeVoice git clone at ~/VibeVoice
        # 3. In the installed vibevoice package (if demo files were included)

        voices_dir = None

        # Try environment variable first
        env_voices_dir = os.environ.get("VIBEVOICE_VOICES_DIR")
        if env_voices_dir:
            voices_dir = Path(env_voices_dir)
            if voices_dir.exists():
                logger.info(f"Using voices from VIBEVOICE_VOICES_DIR: {voices_dir}")

        # Try local VibeVoice clone
        if not voices_dir or not voices_dir.exists():
            local_voices = Path.home() / "VibeVoice" / "demo" / "voices" / "streaming_model"
            if local_voices.exists():
                voices_dir = local_voices
                logger.info(f"Using voices from local VibeVoice clone: {voices_dir}")

        # Try installed package
        if not voices_dir or not voices_dir.exists():
            try:
                import vibevoice
                pkg_voices = Path(vibevoice.__file__).parent.parent / "demo" / "voices" / "streaming_model"
                if pkg_voices.exists():
                    voices_dir = pkg_voices
                    logger.info(f"Using voices from installed package: {voices_dir}")
            except (ImportError, AttributeError):
                pass

        if not voices_dir or not voices_dir.exists():
            logger.error(
                "Voices directory not found. Tried:\n"
                f"  - VIBEVOICE_VOICES_DIR environment variable\n"
                f"  - ~/VibeVoice/demo/voices/streaming_model\n"
                f"  - Installed vibevoice package\n"
            )
            raise RuntimeError(
                "Voice files not found. Please either:\n"
                "  1. Clone VibeVoice to ~/VibeVoice: git clone https://github.com/microsoft/VibeVoice.git ~/VibeVoice\n"
                "  2. Set VIBEVOICE_VOICES_DIR to point to the voices/streaming_model directory"
            )

        # Find matching voice file
        voice_files = list(voices_dir.glob("*.pt"))
        voice_map = {f.stem: f for f in voice_files}

        # Try exact match or partial match
        voice_path = None
        if self.speaker_name in voice_map:
            voice_path = voice_map[self.speaker_name]
        else:
            # Try partial matching (case insensitive)
            speaker_lower = self.speaker_name.lower()
            for name, path in voice_map.items():
                if speaker_lower in name.lower() or name.lower() in speaker_lower:
                    voice_path = path
                    break

        # Default to en-Emma_woman if no match
        if voice_path is None:
            default_name = "en-Emma_woman"
            voice_path = voice_map.get(default_name)
            if voice_path is None and voice_files:
                voice_path = voice_files[0]
            logger.warning(
                f"Speaker '{self.speaker_name}' not found, using {voice_path.stem if voice_path else 'default'}"
            )

        if voice_path is None:
            raise RuntimeError("No voice files found in voices directory")

        logger.info(f"Loading voice sample: {voice_path}")
        target_device = self.device if self.device != "cpu" else "cpu"
        self._voice_sample = torch.load(voice_path, map_location=target_device, weights_only=False)

    # Interface compatibility -------------------------------------------------
    def prepare(self, **kwargs) -> Requirements:
        """Generate audio from text and prepare for streaming."""
        # Ensure model is loaded
        self._ensure_initialized()

        # Extract text from kwargs
        text = kwargs.get("text")

        # Check for transition.target_prompts first (takes precedence over old prompts)
        transition = kwargs.get("transition")
        prompts = None
        if transition and isinstance(transition, dict):
            prompts = transition.get("target_prompts")

        # Fall back to direct prompts if no transition
        if not prompts:
            prompts = kwargs.get("prompts") or []

        if text:
            self._last_text = text
        elif prompts:
            # Use the first prompt's text as the input text
            first = prompts[0]
            if isinstance(first, dict):
                self._last_text = first.get("text")
            else:
                self._last_text = getattr(first, "text", None)

        if not self._last_text:
            logger.warning("No text provided for VibeVoice generation")
            with self._lock:
                self._audio_buffer = _generate_fallback_tone(3.0, self.vibevoice_sample_rate)
                self._position = 0
            return Requirements(input_size=1)

        # Update chunk size if provided
        chunk_size = kwargs.get("chunk_size")
        if chunk_size:
            self.chunk_size = chunk_size

        # Generate audio
        logger.info(f"Generating audio for text: {self._last_text[:100]}...")

        try:
            # Prepare inputs
            inputs = self._processor.process_input_with_cached_prompt(
                text=self._last_text,
                cached_prompt=self._voice_sample,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            # Move tensors to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

            # Generate audio
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=self.cfg_scale,
                    tokenizer=self._processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    all_prefilled_outputs=copy.deepcopy(self._voice_sample),
                )

            # Extract audio (outputs.speech_outputs is a list with one tensor per batch item)
            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                audio_tensor = outputs.speech_outputs[0]  # Shape: (audio_samples,) or (1, audio_samples)

                # Convert to numpy int16 (handle bfloat16 by converting to float32 first)
                if audio_tensor.dtype == torch.bfloat16:
                    audio_tensor = audio_tensor.float()
                audio_np = audio_tensor.cpu().numpy()

                # Flatten if multi-dimensional
                if audio_np.ndim > 1:
                    audio_np = audio_np.flatten()

                # Clamp to [-1, 1] and convert to int16
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = (audio_np * 32767).astype(np.int16)

                # Resample from 24kHz to 48kHz
                audio_48k = _resample_audio(
                    audio_int16,
                    self.vibevoice_sample_rate,
                    self.target_sample_rate,
                )

                with self._lock:
                    self._audio_buffer = audio_48k
                    self._position = 0

                audio_duration = audio_int16.shape[0] / self.vibevoice_sample_rate
                logger.info(f"Generated {audio_duration:.2f}s of audio ({audio_48k.shape[0]} samples @ 48kHz)")
            else:
                logger.error("VibeVoice generation returned no audio")
                with self._lock:
                    self._audio_buffer = _generate_fallback_tone(3.0, self.target_sample_rate)
                    self._position = 0

        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}", exc_info=True)
            with self._lock:
                self._audio_buffer = _generate_fallback_tone(3.0, self.target_sample_rate)
                self._position = 0

        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> torch.Tensor | None:
        """Return the next chunk of audio as a tensor in [-1, 1] range."""
        reset_requested = kwargs.get("init_cache") or kwargs.get("reset_cache")
        if reset_requested:
            self.prepare(**kwargs)

        with self._lock:
            if self._audio_buffer is None or self._audio_buffer.size == 0:
                return None

            if self._position >= self._audio_buffer.shape[0]:
                return None

            end = min(self._position + self.chunk_size, self._audio_buffer.shape[0])
            chunk = self._audio_buffer[self._position : end]
            self._position = end

        # Normalize to [-1, 1] float32 for downstream processing
        chunk_float = np.clip(chunk.astype(np.float32) / 32767.0, -1.0, 1.0)
        return torch.from_numpy(chunk_float)
