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
        from .modules.ltx_core.model.video_vae import TilingConfig
        from .modules.ltx_pipelines.utils import ModelLedger
        from .modules.ltx_pipelines.utils.types import PipelineComponents
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.dtype = dtype
        self.config = config
        
        # Get model paths from config
        model_dir = getattr(config, "model_dir", None)
        if model_dir is None:
            from ....server.models_config import get_models_dir
            models_dir = get_models_dir()
            model_dir = str(models_dir / "ltx2-video")
        
        gemma_root = getattr(config, "gemma_root", None)
        if gemma_root is None:
            from ....server.models_config import get_models_dir
            models_dir = get_models_dir()
            gemma_root = str(models_dir / "google" / "gemma-2-2b-it")
        
        # Initialize model ledger for loading LTX2 components
        start = time.time()
        logger.info("Loading LTX2 models...")
        
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=model_dir,
            gemma_root_path=gemma_root,
            spatial_upsampler_path=None,  # We'll add upsampler support later
            loras=[],
            fp8transformer=False,  # Enable FP8 quantization via config later
        )
        
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=self.device,
        )
        
        # Set up tiling config for VAE decoding
        self.tiling_config = TilingConfig.default()
        
        logger.info(f"LTX2 models loaded in {time.time() - start:.2f}s")
        
        # Cache for model state
        self.text_encoder = None
        self.video_encoder = None
        self.transformer = None
        self.video_decoder = None

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

    def _generate(self, **kwargs) -> torch.Tensor:
        """Internal generation method."""
        from .modules.ltx_core.components.diffusion_steps import EulerDiffusionStep
        from .modules.ltx_core.components.noisers import GaussianNoiser
        from .modules.ltx_core.model.video_vae import decode_video as vae_decode_video
        from .modules.ltx_core.text_encoders.gemma import encode_text
        from .modules.ltx_core.types import VideoPixelShape
        from .modules.ltx_pipelines.utils.constants import (
            DISTILLED_SIGMA_VALUES,
        )
        from .modules.ltx_pipelines.utils.helpers import (
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
        
        torch.cuda.synchronize()
        del text_encoder
        cleanup_memory()
        
        # Load models for generation
        video_encoder = self.model_ledger.video_encoder()
        transformer = self.model_ledger.transformer()
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
        
        # Clean up transformer and encoder
        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()
        
        # Decode video from latents
        logger.info("Decoding video from latents")
        video_decoder = self.model_ledger.video_decoder()
        decoded_video = vae_decode_video(
            video_state.latent,
            video_decoder,
            self.tiling_config
        )
        
        # Clean up decoder
        del video_decoder
        cleanup_memory()
        
        # Convert decoded video iterator to tensor and postprocess
        # LTX2 vae_decode_video returns an iterator of frame chunks
        video_frames = []
        for chunk in decoded_video:
            video_frames.append(chunk)
        
        # Concatenate all chunks along time dimension
        video_tensor = torch.cat(video_frames, dim=1)  # [B, T, C, H, W]
        
        # Convert from BTCHW to THWC format and normalize to [0, 1]
        video_tensor = video_tensor.squeeze(0)  # Remove batch dim: [T, C, H, W]
        video_tensor = video_tensor.permute(0, 2, 3, 1)  # [T, H, W, C]
        
        # LTX2 VAE outputs in range [-1, 1], convert to [0, 1]
        video_tensor = (video_tensor + 1.0) / 2.0
        video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
        
        return video_tensor
