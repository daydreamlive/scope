"""LTX2 text-to-video pipeline implementation."""

import logging
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, PipelineOutput
from .schema import LTX2Config

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class LTX2Pipeline(Pipeline):
    """LTX2 text-to-video generation pipeline.

    This pipeline wraps the LTX2 distilled model for high-quality video generation
    from text prompts. Since LTX2 is not autoregressive, it generates complete videos
    in one shot rather than frame-by-frame.

    Memory Optimization:
    --------------------
    This implementation is optimized for maximum inference speed by keeping all models
    in VRAM:

    1. **Quantization Options for Weights**:
       - **FP8** (default): ~2x reduction, transformer weights ~25GB (requires Ada SM >= 8.9)
       - **NVFP4**: ~4x reduction, transformer weights ~12GB (requires Blackwell SM >= 10.0)
       - **None**: Full precision BF16, transformer weights ~45GB
       Note: **Activations during inference remain in BF16** regardless of weight quantization.

    2. **All Models Cached in VRAM**: Text encoder (~20GB), transformer, and
       video decoder (~3GB) are loaded once during initialization and kept in VRAM.
       This eliminates all model loading overhead between generations.

    3. **PYTORCH_CUDA_ALLOC_CONF**: Set to "expandable_segments:True" in app.py to
       prevent memory fragmentation with quantization.

    4. **VAE Tiling**: Uses TilingConfig for decoder to reduce peak memory during
       video decoding.

    5. **Minimal Defaults**: 33 frames at 512x768 to fit in 96GB VRAM.
       **Activations are the bottleneck**: ~1.5GB per frame at 512x768.

    CRITICAL LIMITATION:
    --------------------
    Quantization only affects weights, not activations. The transformer's intermediate
    activations during denoising consume 60-80GB at higher resolutions regardless of
    weight quantization method.

    Memory Breakdown (96GB GPU with FP8):
    -------------------------------------
    - Text encoder (cached): ~20GB
    - Transformer weights (cached): ~25GB (FP8) / ~12GB (NVFP4) / ~45GB (BF16)
    - Video decoder (cached): ~3GB
    - **Activations during denoising**:
      * 33 frames @ 512x768: ~50GB ✅ Fits
      * 61 frames @ 768x1024: ~90GB ❌ OOM
      * 121 frames @ 1024x1536: ~150GB ❌ OOM

    Reference:
    ----------
    Official LTX-2 documentation:
    https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/README.md
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LTX2Config

    def __init__(
        self,
        config: LTX2Config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize LTX2 pipeline.

        Args:
            config: Pipeline configuration
            device: Target device for inference
            dtype: Data type for model weights
        """
        import sys
        from pathlib import Path

        # Add the modules directory to sys.path so ltx_core and ltx_pipelines can be imported
        modules_dir = Path(__file__).parent / "modules"
        if str(modules_dir) not in sys.path:
            sys.path.insert(0, str(modules_dir))

        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils import ModelLedger
        from ltx_pipelines.utils.types import PipelineComponents

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.dtype = dtype
        self.config = config

        # Get model paths from config
        # Models are downloaded to:
        # - LTX-2/ltx-2-19b-distilled.safetensors (main model checkpoint)
        # - gemma-3-12b-it/ (contains tokenizer and model files)
        checkpoint_path = getattr(config, "checkpoint_path", None)
        if checkpoint_path is None:
            from ....server.models_config import get_models_dir

            models_dir = get_models_dir()
            # checkpoint_path should point to the .safetensors file, not the directory
            checkpoint_path = str(
                models_dir / "LTX-2" / "ltx-2-19b-distilled.safetensors"
            )

        gemma_root = getattr(config, "gemma_root", None)
        if gemma_root is None:
            from ....server.models_config import get_models_dir

            models_dir = get_models_dir()
            gemma_root = str(models_dir / "gemma-3-12b-it")

        # Initialize model ledger for loading LTX2 components
        start = time.time()
        logger.info(f"Loading LTX2 checkpoint from: {checkpoint_path}")
        logger.info(f"Loading Gemma text encoder from: {gemma_root}")

        # Resolve quantization setting
        # Support: Quantization enum, string values, and legacy 'use_fp8' field
        from ..enums import Quantization

        quantization_value = None
        quantization_config = getattr(config, "quantization", None)

        if quantization_config is not None:
            # Handle Quantization enum
            if isinstance(quantization_config, Quantization):
                if quantization_config == Quantization.FP8_E4M3FN:
                    quantization_value = "fp8"
                elif quantization_config == Quantization.NVFP4:
                    quantization_value = "nvfp4"
            # Handle string values (from OmegaConf/dict config)
            elif isinstance(quantization_config, str):
                if quantization_config in ("fp8_e4m3fn", "fp8"):
                    quantization_value = "fp8"
                elif quantization_config == "nvfp4":
                    quantization_value = "nvfp4"
                # None or empty string means no quantization

        # Legacy backwards compatibility with use_fp8 field
        if quantization_value is None:
            use_fp8 = getattr(config, "use_fp8", None)
            if use_fp8 is True:
                quantization_value = "fp8"

        logger.info(f"Creating ModelLedger with quantization={quantization_value}")
        try:
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=None,  # We'll add upsampler support later
                loras=[],
                # Use default DummyRegistry - don't cache state dicts in RAM
                quantization=quantization_value,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ModelLedger: {e}")
            logger.error(f"Make sure model checkpoint is at: {checkpoint_path}")
            logger.error(f"Make sure Gemma text encoder is at: {gemma_root}")
            raise

        # Store resolved quantization for logging
        self._quantization = quantization_value

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

        # Set up tiling config for VAE decoding
        self.tiling_config = TilingConfig.default()

        # Cache all models in VRAM for maximum performance
        # This uses more VRAM (~48GB total) but eliminates all reload overhead
        logger.info("Loading and caching models in VRAM...")
        logger.info("  - Loading text encoder (~20GB)...")
        self._cached_text_encoder = self.model_ledger.text_encoder()
        logger.info("  - Loading transformer (~25GB)...")
        self._cached_transformer = self.model_ledger.transformer()
        logger.info("  - Loading video decoder (~3GB)...")
        self._cached_video_decoder = self.model_ledger.video_decoder()
        logger.info("  - Loading audio decoder...")
        self._cached_audio_decoder = self.model_ledger.audio_decoder()
        logger.info("  - Loading vocoder...")
        self._cached_vocoder = self.model_ledger.vocoder()
        logger.info("All models cached successfully in VRAM")

        logger.info(f"LTX2 models loaded in {time.time() - start:.2f}s")

        # Log quantization status
        if self._quantization == "nvfp4":
            logger.info(
                "NVFP4 quantization: enabled (Blackwell GPU SM >= 10.0, comfy-kitchen)"
            )
            logger.info(
                "NVFP4 provides ~4x memory reduction for transformer weights (~12GB). "
                "Using comfy-kitchen for hardware-accelerated matmul. "
                "Activations during inference are still in BF16 and are the main memory bottleneck. "
                f"At {self.config.height}x{self.config.width} with {self.config.num_frames} frames, "
                "expect ~50-60GB for activations during denoising."
            )
        elif self._quantization == "fp8":
            logger.info("FP8 quantization: enabled")
            logger.warning(
                "FP8 quantization only reduces weight size (~25GB). "
                "Activations during inference are still in BF16 and are the main memory bottleneck. "
                f"At {self.config.height}x{self.config.width} with {self.config.num_frames} frames, "
                "expect ~50-60GB for activations during denoising."
            )
        else:
            logger.info("Quantization: disabled (full precision BF16)")

        # NOTE: This is currently a single-stage pipeline implementation.
        # For even lower VRAM usage, consider implementing a two-stage pipeline:
        # - Stage 1: Generate at lower resolution (512x768) with CFG guidance
        # - Stage 2: Upsample to full resolution (1024x1536) with distilled LoRA
        # See: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py

    def __call__(self, **kwargs) -> dict:
        """Generate video and audio from text prompt.

        Args:
            **kwargs: Generation parameters including:
                - prompts: List of prompt dictionaries with 'text' and 'weight' keys
                - seed: Random seed for generation
                - num_frames: Number of frames to generate (overrides config)
                - frame_rate: Frame rate for video (overrides config)

        Returns:
            Dictionary with "video" key containing PipelineOutput:
                - video: Generated video tensor in THWC format [0, 1] range
                - audio: Generated audio tensor in (channels, samples) format
                - audio_sample_rate: Audio sample rate (24000 Hz)
        """
        return {"video": self._generate(**kwargs)}

    @torch.inference_mode()
    def _generate(self, **kwargs) -> PipelineOutput:
        """Internal generation method."""
        import random

        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import (
            AUDIO_SAMPLE_RATE,
            DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils.helpers import (
            denoise_audio_video,
            euler_denoising_loop,
            simple_denoising_func,
        )

        # Extract parameters
        prompts = kwargs.get("prompts", [{"text": "a beautiful sunset", "weight": 1.0}])
        seed = kwargs.get("seed", kwargs.get("base_seed", 42))
        num_frames = kwargs.get("num_frames", self.config.num_frames)
        frame_rate = kwargs.get("frame_rate", self.config.frame_rate)
        height = kwargs.get("height", self.config.height)
        width = kwargs.get("width", self.config.width)
        randomize_seed = kwargs.get("randomize_seed", self.config.randomize_seed)

        # Randomize seed if enabled (useful for non-autoregressive models like LTX2)
        # This ensures each chunk gets a different seed for varied outputs
        if randomize_seed:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Randomized seed: {seed}")

        # Convert prompts to single text (for now, just use first prompt)
        prompt_text = prompts[0]["text"] if prompts else "a beautiful sunset"

        # Set up generator and components
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Encode text prompt using cached text encoder
        logger.info(f"Encoding prompt: {prompt_text}")
        context_p = encode_text(self._cached_text_encoder, prompts=[prompt_text])[0]
        video_context, audio_context = context_p

        # Use cached transformer for generation
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        # Define denoising loop
        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=self._cached_transformer,
                ),
            )

        # Set up output shape (LTX2 generates at full resolution in one stage for simplicity)
        output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )

        # No image conditioning for now
        conditionings = []

        # Generate video and audio latents
        logger.info(f"Generating {num_frames} frames at {height}x{width}")
        video_state, audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=self.dtype,
            device=self.device,
        )

        # Decode video from latents using cached decoder
        logger.info("Decoding video from latents")

        # Use tiling for VAE decoding to reduce memory usage
        decoded_video = vae_decode_video(
            video_state.latent, self._cached_video_decoder, self.tiling_config
        )

        # Convert decoded video iterator to tensor and postprocess
        # LTX2 vae_decode_video returns an iterator of frame chunks
        video_frames = []
        for chunk in decoded_video:
            video_frames.append(chunk.to(torch.float32))

        # Concatenate all chunks along time dimension -> [T, H, W, C]
        video_tensor = torch.cat(video_frames, dim=0)

        # Normalize from [0, 255] uint8 to [0, 1] float
        video_tensor = torch.clamp(video_tensor / 255.0, 0.0, 1.0)

        # Decode audio from latents using cached audio decoder and vocoder
        # Following the official LTX-2 pipeline:
        # https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/distilled.py
        logger.info("Decoding audio from latents")
        audio_tensor = vae_decode_audio(
            audio_state.latent, self._cached_audio_decoder, self._cached_vocoder
        )
        # audio_tensor shape: (channels, samples) - typically (2, N) for stereo at 24kHz

        return PipelineOutput(
            video=video_tensor,
            audio=audio_tensor,
            audio_sample_rate=AUDIO_SAMPLE_RATE,  # 24000 Hz
        )
