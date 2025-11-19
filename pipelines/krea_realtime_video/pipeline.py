import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState

from lib.schema import Quantization

from ..base.vae import create_vae
from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from .modular_blocks import (
    KreaRealtimeVideoTextBlocks,
    KreaRealtimeVideoVideoBlocks,
)
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]

# Chunk size for video input when operating in video-to-video mode
# TODO: Remove this constant when rebasing on PR 152, along with the prepare() method.
CHUNK_SIZE = 4

WARMUP_RUNS = 3
WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline):
    # Native/default generation mode for this pipeline. In native mode the
    # pipeline runs purely text-to-video and never requires input video unless
    # explicitly requested.
    NATIVE_GENERATION_MODE = "text"

    @classmethod
    def get_defaults(cls) -> dict:
        """Return default parameters for KreaRealtimeVideo pipeline."""
        shared = {
            "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
            "manage_cache": True,
            "base_seed": 42,
            "kv_cache_attention_bias": 0.30,
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
                    "resolution": {"height": 320, "width": 320},
                    "noise_scale": 0.35,
                    "noise_controller": True,
                },
            },
        }

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        compile: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)

        model_config = getattr(config, "model_config", {})
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-14B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})

        # Load generator
        start = time.time()
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            **base_model_kwargs,
        )

        print(f"Loaded diffusion model in {time.time() - start:3f}s")

        for block in generator.model.blocks:
            block.self_attn.fuse_projections()

        # Initialize optional LoRA adapters on the underlying model BEFORE quantization.
        generator.model = self._init_loras(config, generator.model)

        if quantization == Quantization.FP8_E4M3FN:
            # Cast before optional quantization
            generator = generator.to(dtype=dtype)

            start = time.time()

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            # Move to target device during quantization
            # Defaults to using fp8_e4m3fn for both weights and activations
            quantize_(
                generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=device,
            )

            print(f"Quantized diffusion model to fp8 in {time.time() - start:.3f}s")
        else:
            generator = generator.to(device=device, dtype=dtype)

        if compile:
            # Only compile the attention blocks
            for block in generator.model.blocks:
                # Disable fullgraph right now due to issues with RoPE
                block.compile(fullgraph=False)

        # Load text encoder
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

        # Load vae
        start = time.time()
        vae_strategy = getattr(config, "vae_strategy", None)
        vae = create_vae(
            strategy=vae_strategy,
            pipeline_name="krea_realtime_video",
            model_name=base_model_name,
            model_dir=model_dir,
            vae_path=vae_path,
        )
        print(f"Loaded VAE in {time.time() - start:.3f}s")
        # Move VAE to target device and use target dtype
        vae = vae.to(device=device, dtype=dtype)

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        # Separate block graphs for text and video modes share the same
        # underlying modular blocks but avoid requiring video inputs when
        # running in pure text-to-video mode.
        self.blocks_text = KreaRealtimeVideoTextBlocks()
        self.blocks_video = KreaRealtimeVideoVideoBlocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("current_noise_scale", 0.7)

        # Initialize with native mode defaults
        from lib.defaults import get_mode_defaults

        native_defaults = get_mode_defaults(self.__class__)
        self.state.set(
            "height",
            getattr(
                config,
                "height",
                native_defaults.get("resolution", {}).get("height", 320),
            ),
        )
        self.state.set(
            "width",
            getattr(
                config, "width", native_defaults.get("resolution", {}).get("width", 576)
            ),
        )
        self.state.set(
            "base_seed", getattr(config, "seed", native_defaults.get("base_seed", 42))
        )
        self.state.set("manage_cache", native_defaults.get("manage_cache", True))
        self.state.set(
            "kv_cache_attention_bias",
            native_defaults.get("kv_cache_attention_bias", 0.3),
        )
        self.state.set("noise_scale", native_defaults.get("noise_scale"))
        self.state.set("noise_controller", native_defaults.get("noise_controller"))

        start = time.time()
        for _ in range(WARMUP_RUNS):
            self._generate(prompts=WARMUP_PROMPT)

        print(f"Warmed up in {time.time() - start:2f}s")

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

        # Select appropriate block graph based on generation mode. If no mode
        # has been set yet, fall back to the native mode.
        mode = self.state.get("generation_mode") or self.NATIVE_GENERATION_MODE

        # Apply mode-specific defaults for parameters not provided in kwargs
        from lib.defaults import get_mode_defaults

        mode_defaults = get_mode_defaults(self.__class__, mode)
        if "denoising_step_list" not in kwargs and mode_defaults.get("denoising_steps"):
            self.state.set("denoising_step_list", mode_defaults["denoising_steps"])
        if mode == "text":
            self.state.set("noise_scale", None)
        elif "noise_scale" not in kwargs:
            self.state.set("noise_scale", mode_defaults.get("noise_scale"))
        if "noise_controller" not in kwargs:
            self.state.set("noise_controller", mode_defaults.get("noise_controller"))

        blocks = self.blocks_video if mode == "video" else self.blocks_text

        _, self.state = blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["video"])
