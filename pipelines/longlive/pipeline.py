import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState

from ..base.vae import create_vae
from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.lora.strategies.module_targeted_lora import ModuleTargetedLoRAStrategy
from .modular_blocks import LongLiveTextBlocks, LongLiveVideoBlocks
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]

# Chunk size for video input when operating in video-to-video mode
# TODO: Remove this constant when rebasing on PR 152, along with the prepare() method.
CHUNK_SIZE = 4


class LongLivePipeline(Pipeline, LoRAEnabledPipeline):
    # Native/default generation mode for this pipeline. In native mode the
    # pipeline runs purely text-to-video and never requires input video unless
    # explicitly requested.
    NATIVE_GENERATION_MODE = "text"

    @classmethod
    def get_defaults(cls) -> dict:
        """Return default parameters for LongLive pipeline."""
        shared = {
            "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
            "manage_cache": True,
            "base_seed": 42,
        }
        return {
            "native_generation_mode": cls.NATIVE_GENERATION_MODE,
            "modes": {
                "text": {
                    **shared,
                    "resolution": {"height": 320, "width": 576},
                    "noise_scale": None,
                    "noise_controller": None,
                },
                "video": {
                    **shared,
                    "resolution": {"height": 512, "width": 512},
                    "noise_scale": 0.7,
                    "noise_controller": True,
                },
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
        lora_path = getattr(config, "lora_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = getattr(config, "model_config", {})
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

        # Load VAEs - use streamdiffusionv2 strategy for video-to-video mode (streaming-friendly)
        # and longlive strategy for text-to-video mode
        start = time.time()
        vae_strategy = getattr(config, "vae_strategy", None)
        # TODO: Replace this dual-load stopgap with a smarter per-mode VAE
        # selection (e.g. independent strategy overrides or lazy loading) once
        # the video-to-video integration stabilizes. This would help support other implementations (LightVAE) as well.
        vae_text = create_vae(
            strategy=vae_strategy or "longlive",  # Default to longlive for text mode
            pipeline_name="longlive",
            model_dir=model_dir,
        )
        vae_video = create_vae(
            strategy=vae_strategy
            or "streamdiffusionv2",  # Use streamdiffusionv2 for video mode
            pipeline_name="longlive",
            model_dir=model_dir,
            # When reusing the StreamDiffusionV2 VAE inside LongLive, apply
            # LongLive-style latent scaling so that video latents match the
            # distribution expected by the LongLive pipeline.
            apply_longlive_scale=True,
        )
        print(f"Loaded VAEs in {time.time() - start:.3f}s")
        # Move VAEs to target device and use target dtype
        vae_text = vae_text.to(device=device, dtype=dtype)
        vae_video = vae_video.to(device=device, dtype=dtype)

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae_text", vae_text)
        components.add("vae_video", vae_video)
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

        # Initialize state with defaults
        self._initialize_state_with_defaults(self.state, config)

        self.first_call = True

    def prepare(
        self, generation_mode: str | None = None, **kwargs
    ) -> Requirements | None:
        """
        Determine whether this call should consume video input.

        When generation_mode is \"video\", the pipeline requests CHUNK_SIZE
        frames from the FrameProcessor and operates in video-to-video mode
        using the shared video preprocessing and latent blocks. When
        generation_mode is \"text\" (the native mode), no video is requested
        and the pipeline operates in pure text-to-video mode using noise
        latents only.
        """
        # TODO: Remove this method when rebasing on PR 152, along with the CHUNK_SIZE
        # constant. The prepare() functionality will be moved to the base Pipeline class.
        mode = (
            generation_mode
            or kwargs.get("generation_mode")
            or self.NATIVE_GENERATION_MODE
        )
        if mode == "video":
            return Requirements(input_size=CHUNK_SIZE)
        return None

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

        # Select appropriate block graph and VAE based on generation mode. If
        # no mode has been set yet, fall back to the native mode.
        mode = self.state.get("generation_mode") or self.NATIVE_GENERATION_MODE

        # Apply mode-specific defaults for parameters not provided in kwargs
        from lib.defaults import apply_mode_defaults_to_state

        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        blocks = self.blocks_video if mode == "video" else self.blocks_text

        # Select VAE based on mode: streamdiffusionv2 for video mode, longlive for text mode
        if mode == "video":
            self.components.add("vae", self.components.vae_video)
        else:
            self.components.add("vae", self.components.vae_text)

        _, self.state = blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["video"])
