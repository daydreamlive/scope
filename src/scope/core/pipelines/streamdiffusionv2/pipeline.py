import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState
from PIL import Image

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from .components import WanVAEWrapper
from .modular_blocks import StreamDiffusionV2Blocks
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [750, 250]

# Chunk size for streamdiffusionv2
CHUNK_SIZE = 4


class StreamDiffusionV2Pipeline(Pipeline, LoRAEnabledPipeline):
    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,  # Allow extra kwargs
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

        # Load VAE
        start = time.time()
        vae = WanVAEWrapper(model_dir=model_dir)
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

        self.blocks = StreamDiffusionV2Blocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)
        self.state.set("noise_scale", 0.7)
        self.state.set("noise_controller", True)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "seed", 42))

        self.first_call = True

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements:
        return Requirements(input_size=CHUNK_SIZE)

    def __call__(
        self,
        **kwargs,
    ) -> torch.Tensor:
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

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        _, self.state = self.blocks(self.components, self.state)

        return postprocess_chunk(self.state.values["output_video"])
