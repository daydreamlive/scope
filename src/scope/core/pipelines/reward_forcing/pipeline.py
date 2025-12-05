"""Reward-Forcing Pipeline for few-step video generation.

Reward-Forcing is a training methodology that enables high-quality video
generation in just 4 denoising steps by learning from reward signals.

Reference: https://github.com/JaydenLu666/Reward-Forcing
"""

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
from ..schema import RewardForcingConfig
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.vae import WanVAEWrapper
from .modular_blocks import RewardForcingBlocks
from .modules import RewardForcingCausalModel

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Reward-Forcing uses 4-step denoising
DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class RewardForcingPipeline(Pipeline, LoRAEnabledPipeline):
    """Pipeline for Reward-Forcing video generation.

    This pipeline generates videos using a distilled model that can produce
    high-quality results in just 4 denoising steps. It shares the same
    architecture as LongLive but uses differently trained weights.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return RewardForcingConfig

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
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

        # Load generator with EMA sink capable model
        start = time.time()
        generator = WanDiffusionWrapper(
            RewardForcingCausalModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )
        print(f"Loaded Reward-Forcing diffusion model in {time.time() - start:.3f}s")

        # Initialize any user-configured LoRA adapters via shared manager
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

        self.blocks = RewardForcingBlocks()
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

        # Handle compression_alpha updates (no cache reset needed)
        compression_alpha = kwargs.get("compression_alpha")
        if compression_alpha is not None:
            self._update_compression_alpha(compression_alpha)

        # Handle sink_size updates (requires cache reset)
        sink_size = kwargs.get("sink_size")
        if sink_size is not None:
            self._update_sink_size(sink_size)
            # Trigger cache reset when sink_size changes
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        for k, v in kwargs.items():
            self.state.set(k, v)

        # Clear transition from state if not provided to prevent stale transitions
        if "transition" not in kwargs:
            self.state.set("transition", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])

    def _update_compression_alpha(self, alpha: float) -> None:
        """Update compression_alpha on all self-attention layers.

        This can be changed at runtime without reloading the pipeline.
        Higher values (0.999) preserve original context longer.
        Lower values (0.95) reduce error accumulation.

        Args:
            alpha: New compression alpha value (0.0 to 1.0)
        """
        model = self.components.generator.model
        updated_count = 0
        for module in model.modules():
            if hasattr(module, "compression_alpha"):
                module.compression_alpha = alpha
                updated_count += 1
        logger.info(f"Updated compression_alpha to {alpha} on {updated_count} layers")

    def _update_sink_size(self, sink_size: int) -> None:
        """Update sink_size on all self-attention layers.

        This requires a cache reset but NOT a pipeline reload.
        More sink tokens = stronger semantic preservation.

        Args:
            sink_size: Number of frames to keep as sink tokens (1-12)
        """
        model = self.components.generator.model
        updated_count = 0
        for module in model.modules():
            if hasattr(module, "sink_size"):
                module.sink_size = sink_size
                updated_count += 1
        # Also update config if it exists
        if hasattr(self.components, "config") and hasattr(
            self.components.config, "base_model_kwargs"
        ):
            self.components.config.base_model_kwargs["sink_size"] = sink_size
        logger.info(f"Updated sink_size to {sink_size} on {updated_count} layers")
