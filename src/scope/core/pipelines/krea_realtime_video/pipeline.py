import logging
import time

import torch

from ..defaults import INPUT_MODE_TEXT
from ..helpers import build_pipeline_schema
from ..multi_mode import MultiModePipeline
from ..utils import Quantization, calculate_input_size, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from .modular_blocks import KreaRealtimeVideoWorkflow
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]

WARMUP_RUNS = 3
WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(MultiModePipeline, LoRAEnabledPipeline):
    """KreaRealtimeVideo pipeline using declarative MultiModePipeline architecture.

    This pipeline supports both text-to-video and video-to-video generation
    with advanced KV cache attention control for realtime performance. Mode routing
    is handled automatically based on input presence (video input triggers V2V mode).
    Uses nested AutoPipelineBlocks for input-based workflow routing.
    """

    @classmethod
    def get_schema(cls) -> dict:
        """Return schema for Krea Realtime Video pipeline."""
        return build_pipeline_schema(
            pipeline_id="krea-realtime-video",
            name="Krea Realtime Video",
            description="Text-to-video generation with advanced KV cache attention control for realtime performance",
            native_mode=INPUT_MODE_TEXT,
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
                "input_size": calculate_input_size(__file__),
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
        return KreaRealtimeVideoWorkflow()

    @classmethod
    def get_components(cls) -> dict:
        """Declare component requirements for KreaRealtimeVideo pipeline."""
        return {
            "generator": WanDiffusionWrapper,
            "text_encoder": WanTextEncoderWrapper,
            "vae": {
                "text": {"strategy": "krea_realtime_video"},
                "video": {"strategy": "krea_realtime_video"},
            },
        }

    @classmethod
    def get_defaults(cls) -> dict:
        """Return mode-specific defaults for KreaRealtimeVideo pipeline."""
        return {
            "text": {
                "resolution": {"height": 320, "width": 576},
            },
            "video": {
                "resolution": {"height": 320, "width": 320},
                "noise_scale": 0.35,
                "noise_controller": True,
                "input_size": calculate_input_size(__file__),
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

        # Prepare VAE initialization kwargs for lazy loading
        vae_init_kwargs = {
            "model_name": base_model_name,
            "model_dir": model_dir,
            "vae_path": vae_path,
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

        # Warmup runs for JIT optimization
        start = time.time()
        for _ in range(WARMUP_RUNS):
            super().__call__(prompts=WARMUP_PROMPT)

        print(f"Warmed up in {time.time() - start:2f}s")

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
