import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..defaults import GENERATION_MODE_TEXT
from ..helpers import build_pipeline_schema
from ..interface import Pipeline
from ..mode_helpers import UniversalInputModesMixin
from ..utils import load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.lora.strategies.module_targeted_lora import ModuleTargetedLoRAStrategy
from .modular_blocks import LongLiveTextBlocks, LongLiveVideoBlocks
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]

# Chunk size for video input when operating in video-to-video mode
CHUNK_SIZE = 4


class LongLivePipeline(UniversalInputModesMixin, Pipeline, LoRAEnabledPipeline):
    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for LongLive pipeline."""
        return build_pipeline_schema(
            pipeline_id="longlive",
            name="LongLive",
            description="Text-to-video generation optimized for long-form content with efficient recaching",
            native_mode=GENERATION_MODE_TEXT,
            shared={
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "manage_cache": True,
                "base_seed": 42,
            },
            text_overrides={
                "resolution": {"height": 320, "width": 576},
                "noise_scale": None,
                "noise_controller": None,
            },
            video_overrides={
                "resolution": {"height": 512, "width": 512},
                "noise_scale": 0.7,
                "noise_controller": True,
                "input_size": CHUNK_SIZE,
                "vae_strategy": "streamdiffusionv2_longlive_scaled",
            },
        )

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
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

        print(f"Loaded diffusion model in {time.time() - start:.3f}s")

        # Apply LongLive's built-in performance LoRA using the module-targeted strategy.
        # This mirrors the original LongLive behavior and is independent of any
        # additional runtime LoRA strategies managed by LoRAEnabledPipeline.
        if lora_path is not None:
            start = time.time()
            # LongLive's adapter config is passed through unchanged so the
            # module-targeted manager can construct the PEFT config exactly
            # like the original implementation.
            longlive_lora_config = dict(lora_config) if lora_config is not None else {}
            generator.model = ModuleTargetedLoRAStrategy._configure_lora_for_model(
                generator.model,
                model_name=generator_model_name,
                lora_config=longlive_lora_config,
            )
            ModuleTargetedLoRAStrategy._load_lora_checkpoint(generator.model, lora_path)
            print(f"Loaded diffusion LoRA in {time.time() - start:.3f}s")

        # Initialize any additional, user-configured LoRA adapters via shared manager.
        # This is additive and does not replace the original LongLive performance LoRA.
        generator.model = self._init_loras(config, generator.model)

        generator = generator.to(device=device, dtype=dtype)

        start = time.time()
        text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Initialize VAE lazy loading infrastructure
        self._init_vae_lazy_loading(
            device=device,
            dtype=dtype,
            model_dir=model_dir,
        )

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        # Separate block graphs for text and video modes share the same
        # underlying modular blocks but avoid requiring video inputs when
        # running in pure text-to-video mode.
        self.blocks_text = LongLiveTextBlocks()
        self.blocks_video = LongLiveVideoBlocks()
        self.components = components
        self.state = PipelineState()

        # Initialize state with native mode defaults
        from ..defaults import get_mode_config
        from ..helpers import initialize_state_from_config

        native_mode_config = get_mode_config(self.__class__)
        initialize_state_from_config(self.state, config, native_mode_config)

        self.first_call = True

    def __call__(self, **kwargs) -> torch.Tensor:
        if self.first_call:
            self.state.set("init_cache", True)
            self.first_call = False
        else:
            # This will be overriden if the init_cache is passed in kwargs
            self.state.set("init_cache", False)

        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> torch.Tensor:
        # Handle runtime LoRA scale updates before writing into state.
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            # Trigger cache reset on LoRA scale updates if manage_cache is enabled
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        return self._prepare_and_execute_blocks(self.state, self.components, **kwargs)
