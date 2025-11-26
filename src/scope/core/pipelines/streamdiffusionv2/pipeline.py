import logging
import time

import torch

from ..defaults import GENERATION_MODE_VIDEO
from ..helpers import build_pipeline_schema
from ..multi_mode import MultiModePipeline
from ..utils import load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from .modular_blocks import StreamDiffusionV2UnifiedWorkflow
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [750, 250]
TEXT_DENOISING_STEP_LIST = [1000, 250]
# Chunk size for streamdiffusionv2
CHUNK_SIZE = 4


class StreamDiffusionV2Pipeline(MultiModePipeline, LoRAEnabledPipeline):
    """StreamDiffusionV2 pipeline using declarative MultiModePipeline architecture.

    This pipeline supports both text-to-video and video-to-video generation
    with efficient streaming and temporal consistency. It uses the new declarative
    pattern where the pipeline simply declares its capabilities via class methods.
    """

    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for StreamDiffusionV2 pipeline."""
        return build_pipeline_schema(
            pipeline_id="streamdiffusionv2",
            name="StreamDiffusion V2",
            description="Video-to-video generation with temporal consistency and efficient streaming",
            native_mode=GENERATION_MODE_VIDEO,
            shared={
                "resolution": {"height": 512, "width": 512},
                "manage_cache": True,
                "base_seed": 42,
            },
            text_overrides={
                "denoising_steps": TEXT_DENOISING_STEP_LIST,
                "noise_scale": None,
                "noise_controller": None,
            },
            video_overrides={
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "noise_scale": 0.7,
                "noise_controller": True,
                "input_size": CHUNK_SIZE,
            },
        )

    @classmethod
    def get_blocks(cls):
        """Return unified workflow for StreamDiffusionV2 pipeline.

        This returns a single SequentialPipelineBlocks that handles both
        text-to-video and video-to-video modes through conditional block execution.
        """
        return StreamDiffusionV2UnifiedWorkflow()

    @classmethod
    def get_components(cls) -> dict:
        """Declare component requirements for StreamDiffusionV2 pipeline."""
        return {
            "generator": WanDiffusionWrapper,
            "text_encoder": WanTextEncoderWrapper,
            "vae": {
                "text": {"strategy": "streamdiffusionv2"},
                "video": {"strategy": "streamdiffusionv2"},
            },
        }

    @classmethod
    def get_defaults(cls) -> dict:
        """Return mode-specific defaults for StreamDiffusionV2 pipeline."""
        return {
            "text": {
                "denoising_steps": TEXT_DENOISING_STEP_LIST,
            },
            "video": {
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "noise_scale": 0.7,
                "noise_controller": True,
                "input_size": CHUNK_SIZE,
            },
        }

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )

        # Load generator
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        print(f"Loaded diffusion model in {time.time() - start:.3f}s")

        # Initialize optional LoRA adapters on the underlying model.
        generator.model = self._init_loras(config, generator.model)

        generator = generator.to(device=device, dtype=dtype)

        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:.3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Prepare VAE initialization kwargs for lazy loading
        vae_init_kwargs = {
            "model_dir": model_dir,
        }

        # Convert model_config to dict for MultiModePipeline
        from omegaconf import OmegaConf

        model_config_dict = OmegaConf.to_container(model_config, resolve=True)

        # Initialize via MultiModePipeline
        super().__init__(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config_dict,
            device=device,
            dtype=dtype,
            vae_init_kwargs=vae_init_kwargs,
        )

    def __call__(self, **kwargs) -> torch.Tensor:
        """Execute pipeline with LoRA handling.

        Args:
            **kwargs: Generation parameters

        Returns:
            Post-processed output tensor
        """
        # Handle runtime LoRA scale updates before execution
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            # Trigger cache reset on LoRA scale updates if manage_cache is enabled
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        # Call parent implementation which handles the rest
        return super().__call__(**kwargs)
