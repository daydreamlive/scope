import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..interface import Pipeline
from ..process import postprocess_chunk
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from .components import WanVAEWrapper
from .modular_blocks import KreaRealtimeVideoBlocks
from .modules.causal_model import CausalWanModel

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]
# This default value < 1.0 will trigger torch.compile in the flex_attention code path
# during warmup
DEFAULT_KV_CACHE_ATTENTION_BIAS = 0.3

WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline):
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

        # Load vae
        start = time.time()
        vae = WanVAEWrapper(
            model_name=base_model_name, model_dir=model_dir, vae_path=vae_path
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

        self.blocks = KreaRealtimeVideoBlocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", DEFAULT_KV_CACHE_ATTENTION_BIAS)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "seed", 42))

        # Warm-up: Run enough iterations to fill the KV cache completely.
        # This ensures torch.compile compiles the flex_attention kernel at the
        # steady-state cache size, avoiding recompilation during actual streaming.
        #
        # Cache fills at: num_frame_per_block frames per iteration
        # Cache capacity: local_attn_size frames
        # Iterations needed: ceil(local_attn_size / num_frame_per_block) + 1
        #   (+1 to exercise the "cache full with eviction" path)
        local_attn_size = getattr(model_config, "local_attn_size", 6)
        num_frame_per_block = getattr(model_config, "num_frame_per_block", 3)
        warmup_runs = (local_attn_size // num_frame_per_block) + 1

        start = time.time()
        for i in range(warmup_runs):
            self._generate(
                prompts=WARMUP_PROMPT,
                init_cache=(i == 0),  # Only init on first run, then accumulate
            )

        print(f"Warmed up ({warmup_runs} runs) in {time.time() - start:.2f}s")

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

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
