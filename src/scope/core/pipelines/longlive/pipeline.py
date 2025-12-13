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
from ..schema import LongLiveConfig
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.lora.strategies.module_targeted_lora import ModuleTargetedLoRAStrategy
from ..wan2_1.vae import WanVAEWrapper
from .modular_blocks import LongLiveBlocks
from .modules.causal_model import CausalWanModel
from .modules.causal_vace_model import CausalVaceWanModel

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class LongLivePipeline(Pipeline, LoRAEnabledPipeline):
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
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        lora_path = getattr(config, "lora_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vace_path = getattr(config, "vace_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )
        lora_config = getattr(model_config, "adapter", {})

        # Load generator (with optional VACE support)
        # Strategy: Load LongLive base, add VACE-specific weights, then apply LoRA
        # (VACE loaded before LoRA to avoid PEFT wrapper unwrapping issues)
        start = time.time()
        model_class = CausalVaceWanModel if vace_path is not None else CausalWanModel

        # VACE configuration: vace_in_dim determines input channel count
        # - vace_in_dim=96: R2V mode (16 base * 6 for masking)
        # - vace_in_dim=16: Depth mode (no masking)
        # Default to 96 if not specified (R2V mode)
        if vace_path is not None:
            base_model_kwargs = dict(base_model_kwargs) if base_model_kwargs else {}
            if "vace_in_dim" not in base_model_kwargs:
                base_model_kwargs["vace_in_dim"] = 96

        generator = WanDiffusionWrapper(
            model_class,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        model_type_str = "VACE-enabled" if vace_path is not None else "standard"
        print(f"Loaded {model_type_str} diffusion model in {time.time() - start:.3f}s")

        # Load VACE-specific weights before LoRA application to avoid PEFT wrapper unwrapping issues
        # VACE weights (vace_blocks.*, vace_patch_embedding.*) are independent of LoRA-modified layers
        if vace_path is not None:
            start = time.time()
            from .vace_weight_loader import load_vace_weights_only

            load_vace_weights_only(generator.model, vace_path)
            print(f"Loaded VACE conditioning weights in {time.time() - start:.3f}s")

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
        print(f"Loaded text encoder in {time.time() - start:3f}s")
        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)

        # Load VAE using unified WanVAEWrapper
        start = time.time()
        vae = WanVAEWrapper(model_dir=model_dir, model_name=base_model_name)
        print(f"Loaded VAE in {time.time() - start:.3f}s")
        # Move VAE to target device and use target dtype
        vae = vae.to(device=device, dtype=dtype)

        # Load separate vace_vae for VACE encoding (depth mode)
        # This ensures depth encoding doesn't corrupt main VAE's autoregressive cache
        vace_vae = None
        if vace_path is not None:
            start = time.time()
            vace_vae = WanVAEWrapper(model_dir=model_dir, model_name=base_model_name)
            print(
                f"Loaded VACE VAE (separate instance for depth encoding) in {time.time() - start:.3f}s"
            )
            vace_vae = vace_vae.to(device=device, dtype=dtype)

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("vae", vae)
        if vace_vae is not None:
            components.add("vace_vae", vace_vae)
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(
            device=device,
            dtype=dtype,
        )
        components.add("embedding_blender", embedding_blender)

        self.blocks = LongLiveBlocks()
        self.components = components
        self.state = PipelineState()
        # These need to be set right now because InputParam.default on the blocks
        # does not work properly
        self.state.set("current_start_frame", 0)
        self.state.set("manage_cache", True)
        self.state.set("kv_cache_attention_bias", 1.0)

        self.state.set("height", config.height)
        self.state.set("width", config.width)
        self.state.set("base_seed", getattr(config, "seed", 42))

        self.first_call = True
        self.last_mode = None  # Track mode for transition detection

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode."""
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> torch.Tensor:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
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

        # Log ref_images before setting into state
        if "ref_images" in kwargs:
            ref_images = kwargs.get("ref_images", [])
            logger.info(
                f"LongLivePipeline._generate: Setting ref_images into state: "
                f"count={len(ref_images) if ref_images else 0}, "
                f"paths={ref_images if ref_images else 'None'}"
            )
        else:
            logger.debug("LongLivePipeline._generate: No ref_images in kwargs")

        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        # Clear ref_images from state if not provided to prevent encoding on chunks where they weren't sent
        if "ref_images" not in kwargs:
            self.state.set("ref_images", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
