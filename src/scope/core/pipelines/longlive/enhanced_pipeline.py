# Enhanced LongLive pipeline with FreSca and TSR support
import logging
import time
from typing import TYPE_CHECKING

import torch
from diffusers.modular_pipelines import PipelineState

from ..blending import EmbeddingBlender
from ..components import ComponentsManager
from ..defaults import (
    apply_mode_defaults_to_state,
    handle_mode_transition,
    prepare_for_mode,
    resolve_input_mode,
)
from ..interface import Pipeline, Requirements
from ..process import postprocess_chunk
from ..utils import Quantization, load_model_config, validate_resolution
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.lora.strategies.module_targeted_lora import ModuleTargetedLoRAStrategy
from ..wan2_1.vace import VACEEnabledPipeline
from ..wan2_1.vae import create_vae
from .enhanced_modular_blocks import EnhancedLongLiveBlocks
from .schema import LongLiveConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class EnhancedLongLivePipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    """
    Enhanced LongLive pipeline with FreSca and TSR support.

    This pipeline adds two enhancement techniques:
    - FreSca: Frequency-selective scaling for detail enhancement
    - TSR: Temporal score rescaling for improved temporal coherence

    Usage:
        pipeline = EnhancedLongLivePipeline(config, device=device)
        output = pipeline(
            prompts=prompts,
            enable_fresca=True,
            fresca_scale_high=1.2,
            enable_tsr=True,
        )
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LongLiveConfig

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from .modules.causal_model import CausalWanModel

        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=16,
        )

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

        start = time.time()

        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        generator.model = self._init_vace(
            config, generator.model, device=device, dtype=dtype
        )

        if lora_path is not None:
            start = time.time()
            longlive_lora_config = dict(lora_config) if lora_config is not None else {}
            generator.model = ModuleTargetedLoRAStrategy._configure_lora_for_model(
                generator.model,
                model_name=generator_model_name,
                lora_config=longlive_lora_config,
            )
            ModuleTargetedLoRAStrategy._load_lora_checkpoint(generator.model, lora_path)
            print(f"Loaded diffusion LoRA in {time.time() - start:.3f}s")

        generator.model = self._init_loras(config, generator.model)

        if quantization == Quantization.FP8_E4M3FN:
            generator = generator.to(dtype=dtype)

            start = time.time()

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

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
        print(f"Loaded text encoder in {time.time() - start:3f}s")
        text_encoder = text_encoder.to(device=device)

        vae_type = getattr(config, "vae_type", "wan")
        start = time.time()
        vae = create_vae(
            model_dir=model_dir, model_name=base_model_name, vae_type=vae_type
        )
        print(f"Loaded VAE (type={vae_type}) in {time.time() - start:.3f}s")
        vae = vae.to(device=device, dtype=dtype)

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

        # Use enhanced blocks
        self.blocks = EnhancedLongLiveBlocks()
        self.components = components
        self.state = PipelineState()

        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "seed", 42))

        # Default enhancement settings (disabled by default)
        self.state.set("enable_fresca", False)
        self.state.set("enable_tsr", False)

        self.first_call = True
        self.last_mode = None

    def prepare(self, **kwargs) -> Requirements | None:
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> torch.Tensor:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> torch.Tensor:
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        for k, v in kwargs.items():
            self.state.set(k, v)

        if "transition" not in kwargs:
            self.state.set("transition", None)

        if "video" not in kwargs:
            self.state.set("video", None)

        if "vace_ref_images" not in kwargs:
            self.state.set("vace_ref_images", None)

        if "first_frame_image" not in kwargs:
            self.state.set("first_frame_image", None)
        if "last_frame_image" not in kwargs:
            self.state.set("last_frame_image", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
