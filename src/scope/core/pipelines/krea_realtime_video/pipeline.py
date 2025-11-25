import logging
import time

import torch

from ..defaults import GENERATION_MODE_TEXT
from ..helpers import build_pipeline_schema
from ..interface import Pipeline
from ..mode_helpers import UniversalInputModesMixin
from ..utils import Quantization, load_model_config
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
CHUNK_SIZE = 4

WARMUP_RUNS = 3
WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(
    UniversalInputModesMixin, Pipeline, LoRAEnabledPipeline
):
    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for Krea Realtime Video pipeline."""
        return build_pipeline_schema(
            pipeline_id="krea-realtime-video",
            name="Krea Realtime Video",
            description="Text-to-video generation with advanced KV cache attention control for realtime performance",
            native_mode=GENERATION_MODE_TEXT,
            shared={
                "denoising_steps": DEFAULT_DENOISING_STEP_LIST,
                "manage_cache": True,
                "base_seed": 42,
                "kv_cache_attention_bias": 0.30,
            },
            text_overrides={
                "resolution": {"height": 320, "width": 576},
                "noise_scale": None,
                "noise_controller": None,
            },
            video_overrides={
                "resolution": {"height": 320, "width": 320},
                "noise_scale": 0.35,
                "noise_controller": True,
            },
        )

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

        model_config = load_model_config(config, __file__)
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

        # Initialize VAE lazy loading infrastructure
        self._init_vae_lazy_loading(
            device=device,
            dtype=dtype,
            model_name=base_model_name,
            model_dir=model_dir,
            vae_path=vae_path,
        )

        # Initialize pipeline state and components using shared helper
        self._initialize_pipeline_state(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            blocks_text=KreaRealtimeVideoTextBlocks(),
            blocks_video=KreaRealtimeVideoVideoBlocks(),
            model_config=model_config,
            device=device,
            dtype=dtype,
        )

        # Warmup runs for JIT optimization
        start = time.time()
        for _ in range(WARMUP_RUNS):
            self._generate(prompts=WARMUP_PROMPT)

        print(f"Warmed up in {time.time() - start:2f}s")

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
