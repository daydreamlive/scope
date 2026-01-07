"""LTX2 text-to-video pipeline implementation."""

import logging
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline
from ..process import postprocess_chunk
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
    This implementation follows the official LTX-2 memory optimization guidelines:

    1. **FP8 Quantization for Weights Only**: Enabled by default (use_fp8=True) to
       reduce transformer weights from ~45GB to ~25GB. However, **activations during
       inference remain in BF16** and are the main memory bottleneck.

    2. **Aggressive Model Cleanup**: Models are loaded on-demand and immediately freed
       after use, with explicit garbage collection.

    3. **PYTORCH_CUDA_ALLOC_CONF**: Set to "expandable_segments:True" in app.py to
       prevent memory fragmentation with FP8 quantization.

    4. **VAE Tiling**: Uses TilingConfig for decoder to reduce peak memory during
       video decoding.

    5. **No Unnecessary Models**: Video encoder is only loaded for i2v conditioning.

    6. **Minimal Defaults**: 33 frames at 512x768 to fit in 96GB VRAM.
       **Activations are the bottleneck**: ~1.5GB per frame at 512x768.

    CRITICAL LIMITATION:
    --------------------
    Unlike other pipelines that use torchao FP8 quantization for both weights AND
    activations, LTX2's custom FP8 only quantizes weights. This means the transformer's
    intermediate activations during denoising consume 60-80GB at higher resolutions.

    Memory Breakdown (96GB GPU):
    ----------------------------
    - Transformer weights (FP8): ~25GB
    - Text encoder (temp): ~20GB (freed before generation)
    - Decoder (temp): ~3GB (freed after decoding)
    - **Activations during denoising**:
      * 33 frames @ 512x768: ~50GB ✅ Fits
      * 61 frames @ 768x1024: ~90GB ❌ OOM
      * 121 frames @ 1024x1536: ~150GB ❌ OOM

    Peak VRAM = Model (25GB) + Activations (resolution × frames dependent)

    For higher quality, you need:
    - A GPU with 141GB+ VRAM (H100)
    - OR the two-stage pipeline (low-res generation + upsampling)
    - OR generate shorter/lower-res videos

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
            checkpoint_path = str(models_dir / "LTX-2" / "ltx-2-19b-distilled.safetensors")

        gemma_root = getattr(config, "gemma_root", None)
        if gemma_root is None:
            from ....server.models_config import get_models_dir
            models_dir = get_models_dir()
            gemma_root = str(models_dir / "gemma-3-12b-it")

        # Initialize model ledger for loading LTX2 components
        start = time.time()
        logger.info(f"Loading LTX2 checkpoint from: {checkpoint_path}")
        logger.info(f"Loading Gemma text encoder from: {gemma_root}")

        # Enable FP8 quantization by default to reduce VRAM usage
        # According to official LTX-2 docs, this significantly reduces memory footprint
        fp8_enabled = getattr(config, "use_fp8", True)

        try:
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
                spatial_upsampler_path=None,  # We'll add upsampler support later
                loras=[],
                # Use default DummyRegistry - don't cache state dicts in RAM
                fp8transformer=fp8_enabled,  # FP8 significantly reduces VRAM usage
            )
        except Exception as e:
            logger.error(f"Failed to initialize ModelLedger: {e}")
            logger.error(f"Make sure model checkpoint is at: {checkpoint_path}")
            logger.error(f"Make sure Gemma text encoder is at: {gemma_root}")
            raise

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )

        # Set up tiling config for VAE decoding
        self.tiling_config = TilingConfig.default()

        logger.info(f"LTX2 models loaded in {time.time() - start:.2f}s")
        logger.info(f"FP8 quantization: {'enabled' if fp8_enabled else 'disabled'}")
        if fp8_enabled:
            logger.warning(
                "FP8 quantization only reduces weight size (~25GB). "
                "Activations during inference are still in BF16 and are the main memory bottleneck. "
                f"At {self.config.height}x{self.config.width} with {self.config.num_frames} frames, "
                "expect ~50-60GB for activations during denoising."
            )

        # NOTE: This is currently a single-stage pipeline implementation.
        # For even lower VRAM usage, consider implementing a two-stage pipeline:
        # - Stage 1: Generate at lower resolution (512x768) with CFG guidance
        # - Stage 2: Upsample to full resolution (1024x1536) with distilled LoRA
        # See: https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py

    def __call__(self, **kwargs) -> torch.Tensor:
        """Generate video from text prompt.

        Args:
            **kwargs: Generation parameters including:
                - prompts: List of prompt dictionaries with 'text' and 'weight' keys
                - seed: Random seed for generation
                - num_frames: Number of frames to generate (overrides config)
                - frame_rate: Frame rate for video (overrides config)

        Returns:
            Generated video tensor in THWC format [0, 1] range
        """
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, **kwargs) -> torch.Tensor:
        """Internal generation method."""
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import (
            DISTILLED_SIGMA_VALUES,
        )
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
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

        # Convert prompts to single text (for now, just use first prompt)
        prompt_text = prompts[0]["text"] if prompts else "a beautiful sunset"

        # Set up generator and components
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # Encode text prompt
        logger.info(f"Encoding prompt: {prompt_text}")
        text_encoder = self.model_ledger.text_encoder()
        context_p = encode_text(text_encoder, prompts=[prompt_text])[0]
        video_context, audio_context = context_p

        # CRITICAL: Aggressively free text encoder memory
        # The Gemma text encoder is HUGE (~20GB for 5 shards) and must be freed immediately
        del text_encoder
        torch.cuda.synchronize()
        cleanup_memory()
        logger.info("Text encoder freed from VRAM")

        # Load transformer for generation
        # NOTE: We do NOT load video_encoder here because it's only needed for i2v conditioning
        # Loading it wastes ~8GB of VRAM unnecessarily
        transformer = self.model_ledger.transformer()
        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        logger.info(f"Loaded transformer for generation")

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
                    transformer=transformer,
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

        # CRITICAL: Clean up transformer immediately after generation
        # According to official docs, this is essential for memory management
        del transformer
        torch.cuda.synchronize()
        cleanup_memory()
        logger.info("Transformer freed from VRAM")

        # Decode video from latents
        logger.info("Decoding video from latents")
        video_decoder = self.model_ledger.video_decoder()

        # Use tiling for VAE decoding to reduce memory usage
        # According to official docs, tiling is essential for memory efficiency
        decoded_video = vae_decode_video(
            video_state.latent,
            video_decoder,
            self.tiling_config
        )

        # CRITICAL: Clean up decoder immediately after use
        del video_decoder
        torch.cuda.synchronize()
        cleanup_memory()
        logger.info("Video decoder freed from VRAM")

        # Convert decoded video iterator to tensor and postprocess
        # LTX2 vae_decode_video returns an iterator of frame chunks
        video_frames = []
        for chunk in decoded_video:
            video_frames.append(chunk.to(torch.float32))

        # Concatenate all chunks along time dimension -> [T, H, W, C]
        video_tensor = torch.cat(video_frames, dim=0)

        # Normalize from [0, 255] uint8 to [0, 1] float
        video_tensor = torch.clamp(video_tensor / 255.0, 0.0, 1.0)

        return video_tensor
