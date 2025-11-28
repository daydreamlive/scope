"""LongLive pipeline implementation using MultiModePipeline architecture.

This is the new declarative implementation of LongLive pipeline that uses
the MultiModePipeline base class for simplified configuration.
"""

import logging
import time

import torch

from ..defaults import INPUT_MODE_TEXT
from ..helpers import build_pipeline_schema
from ..multi_mode import MultiModePipeline
from ..utils import calculate_input_size, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.lora.strategies.module_targeted_lora import ModuleTargetedLoRAStrategy
from .modular_blocks import LongLiveWorkflow
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class LongLivePipeline(MultiModePipeline, LoRAEnabledPipeline):
    """LongLive pipeline using declarative MultiModePipeline architecture.

    This pipeline supports both text-to-video and video-to-video generation
    with efficient recaching for long-form content. Mode routing is handled
    automatically based on input presence (video input triggers V2V mode).
    Uses nested AutoPipelineBlocks for input-based workflow routing.
    """

    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for LongLive pipeline."""
        return build_pipeline_schema(
            pipeline_id="longlive",
            name="LongLive",
            description="Text-to-video generation optimized for long-form content with efficient recaching",
            native_mode=INPUT_MODE_TEXT,
            shared={
                "manage_cache": True,
                "base_seed": 42,
                "default_temporal_interpolation_method": "slerp",
                "default_temporal_interpolation_steps": 0,
            },
            text_overrides={
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "resolution": {"height": 320, "width": 576},
                "noise_scale": None,
                "noise_controller": None,
                "default_prompt": "A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day.",
            },
            video_overrides={
                "denoising_steps": [1000, 750],
                "resolution": {"height": 512, "width": 512},
                "noise_scale": 0.7,
                "noise_controller": True,
                "input_size": calculate_input_size(__file__),
                "vae_strategy": "streamdiffusionv2_longlive_scaled",
                "default_prompt": "A 3D animated scene. A **panda** is sitting at a desk.",
            },
        )

    @classmethod
    def get_blocks(cls):
        """Return single workflow with nested AutoPipelineBlocks routing.

        Returns a unified workflow that uses nested AutoPipelineBlocks
        (AutoPreprocessVideoBlock and AutoPrepareLatentsBlock) for automatic
        routing based on input presence. Routes to V2V when 'video' input is
        provided, otherwise uses T2V latent preparation.
        """
        return LongLiveWorkflow()

    @classmethod
    def get_components(cls) -> dict:
        """Declare component requirements for LongLive pipeline."""
        return {
            "generator": WanDiffusionWrapper,
            "text_encoder": WanTextEncoderWrapper,
            "vae": {
                "text": {"strategy": "longlive"},
                "video": {"strategy": "streamdiffusionv2_longlive_scaled"},
            },
        }

    @classmethod
    def get_defaults(cls) -> dict:
        """Return mode-specific defaults for LongLive pipeline."""
        return {
            "text": {
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "resolution": {"height": 320, "width": 576},
            },
            "video": {
                "denoising_steps": [1000, 750],
                "resolution": {"height": 512, "width": 512},
                "noise_scale": 0.7,
                "noise_controller": True,
                "input_size": calculate_input_size(__file__),
            },
        }

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize LongLive pipeline.

        Args:
            config: Configuration object with model paths and settings
            device: Target device for computation
            dtype: Target dtype for tensors
        """
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        lora_path = getattr(config, "lora_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )
        lora_config = getattr(model_config, "adapter", {})

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

        print(
            f"LongLivePipeline.__init__: Loaded diffusion model in {time.time() - start:.3f}s"
        )

        # Apply LongLive's built-in performance LoRA using the module-targeted strategy.
        if lora_path is not None:
            start = time.time()
            longlive_lora_config = dict(lora_config) if lora_config is not None else {}
            generator.model = ModuleTargetedLoRAStrategy._configure_lora_for_model(
                generator.model,
                model_name=generator_model_name,
                lora_config=longlive_lora_config,
            )
            ModuleTargetedLoRAStrategy._load_lora_checkpoint(generator.model, lora_path)
            print(
                f"LongLivePipeline.__init__: Loaded diffusion LoRA in {time.time() - start:.3f}s"
            )

        # Initialize any additional, user-configured LoRA adapters via shared manager.
        generator.model = self._init_loras(config, generator.model)

        generator = generator.to(device=device, dtype=dtype)

        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(
            f"LongLivePipeline.__init__: Loaded text encoder in {time.time() - start:.3f}s"
        )
        text_encoder = text_encoder.to(device=device)

        # Prepare VAE initialization kwargs for lazy loading
        vae_init_kwargs = {
            "model_dir": model_dir,
        }

        # Convert model_config (OmegaConf) to dict for MultiModePipeline
        # This includes all properties from model.yaml that blocks may need
        # (e.g., max_rope_freq_table_seq_len, num_frame_per_block, etc.)
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
