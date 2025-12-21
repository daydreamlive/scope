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
from ..wan2_1.vace import VACEEnabledPipeline
from ..wan2_1.vae import WanVAEWrapper
from .modular_blocks import LongLiveBlocks
from .modules.causal_model import CausalWanModel

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]


class LongLivePipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
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

        # Load generator with VACE support via upfront loading
        # Strategy: Load LongLive base, wrap with VACE (if enabled), add VACE weights, then apply LoRA
        # (VACE loaded before LoRA to ensure correct wrapper ordering)
        start = time.time()

        # Always create base CausalWanModel first
        generator = WanDiffusionWrapper(
            CausalWanModel,
            model_name=base_model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            generator_model_name=generator_model_name,
            **base_model_kwargs,
        )

        # Apply VACE wrapper if vace_path is configured (upfront loading)
        # This must happen before LoRA to get correct ordering: LoRA -> VACE -> Base
        generator.model = self._init_vace(
            config, generator.model, device=device, dtype=dtype
        )

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

        # Lazy loading infrastructure for VACE
        self.vace_path = vace_path
        self.vace_enabled = False
        self.vace_in_dim = (
            base_model_kwargs.get("vace_in_dim", 96) if base_model_kwargs else 96
        )
        self._stored_model_dir = model_dir
        self._stored_base_model_name = base_model_name
        self._stored_device = device
        self._stored_dtype = dtype

    def _needs_vace(self, kwargs: dict) -> bool:
        """_needs_vace: Check if VACE is needed for this generation call.

        VACE is needed when:
        - ref_images is present and non-empty, OR
        - input_frames is present (for depth/flow/pose conditioning)

        Args:
            kwargs: Generation kwargs

        Returns:
            bool: True if VACE is needed
        """
        ref_images = kwargs.get("vace_ref_images")
        input_frames = kwargs.get("vace_input_frames")

        has_ref_images = ref_images is not None and len(ref_images) > 0
        has_input_frames = input_frames is not None

        return has_ref_images or has_input_frames

    def _enable_vace(self):
        """_enable_vace: Enable VACE by wrapping model and loading weights.

        This is called lazily when VACE is first needed (ref_images or input_frames provided).

        Model Composition Strategy:
        Supports two scenarios:
        1. PEFT-wrapped model (LongLive LoRA or runtime PEFT LoRAs active):
           - Save LoRA adapter configuration and weights
           - Unwrap PEFT (without merging - keep base weights pure)
           - Wrap pure base model with VACE and load VACE weights
           - Reapply LoRA adapter structure on top of VACE
           - Restore LoRA adapter weights
           This produces: PeftModel(CausalVaceWanModel(CausalWanModel))

        2. Non-PEFT model (permanent merge LoRAs or no LoRAs):
           - Wrap base model with VACE directly and load VACE weights
           This produces: CausalVaceWanModel(CausalWanModel)

        In both cases, loads separate VACE VAE for encoding.

        The PEFT unwrap/rewrap ensures correct ordering: LoRA must be on top of VACE,
        not underneath it, to avoid the model devolving into noise.

        Raises:
            RuntimeError: If VACE is needed but vace_path is None
        """
        if self.vace_enabled:
            return

        if self.vace_path is None:
            raise RuntimeError(
                "_enable_vace: VACE is required but vace_path is None. "
                "This should not happen - VACE should be configured during pipeline initialization."
            )

        logger.info(
            f"_enable_vace: Lazy loading VACE support (vace_in_dim={self.vace_in_dim})"
        )

        current_model = self.components.generator.model

        # Check if model is PEFT-wrapped (LongLive LoRA or runtime PEFT LoRAs)
        is_peft_wrapped = hasattr(current_model, "peft_config")

        if is_peft_wrapped:
            from peft import (
                get_peft_model,
                get_peft_model_state_dict,
                set_peft_model_state_dict,
            )

            logger.info("_enable_vace: Model is PEFT-wrapped, will preserve LoRA state")

            # Step 1: Save LoRA adapter configuration and weights
            adapter_state_dict = get_peft_model_state_dict(current_model)
            peft_config = dict(current_model.peft_config)
            active_adapters = (
                list(current_model.active_adapters)
                if hasattr(current_model, "active_adapters")
                else None
            )
            logger.info(
                f"_enable_vace: Saved {len(adapter_state_dict)} LoRA parameters "
                f"from adapters: {list(peft_config.keys())}"
            )

            # Step 2: Unwrap PEFT without merging (keep base weights pure)
            start = time.time()
            base_model = current_model.unload()
            logger.info(
                f"_enable_vace: Unwrapped PEFT (no merge) in {time.time() - start:.3f}s"
            )
        else:
            logger.info(
                "_enable_vace: Model is not PEFT-wrapped (LoRAs permanently merged or none loaded)"
            )
            base_model = current_model

        # Step 3: Wrap with VACE and load weights
        start = time.time()
        vace_wrapped_model = CausalVaceWanModel(
            base_model, vace_in_dim=self.vace_in_dim
        )
        vace_wrapped_model.vace_patch_embedding.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        vace_wrapped_model.vace_blocks.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        logger.info(f"_enable_vace: Wrapped with VACE in {time.time() - start:.3f}s")

        start = time.time()
        from ..wan2_1.vace import load_vace_weights_only

        load_vace_weights_only(vace_wrapped_model, self.vace_path)
        logger.info(f"_enable_vace: Loaded VACE weights in {time.time() - start:.3f}s")

        # Step 4 & 5: Reapply LoRA (only if model was PEFT-wrapped)
        if is_peft_wrapped:
            start = time.time()
            for adapter_name, config in peft_config.items():
                vace_wrapped_model = get_peft_model(
                    vace_wrapped_model, config, adapter_name=adapter_name
                )
                logger.info(f"_enable_vace: Reapplied PEFT adapter '{adapter_name}'")

            # Restore LoRA adapter weights
            set_peft_model_state_dict(vace_wrapped_model, adapter_state_dict)
            logger.info(
                f"_enable_vace: Restored LoRA weights in {time.time() - start:.3f}s"
            )

            # Restore active adapters
            if active_adapters is not None:
                adapter_name = (
                    active_adapters[0] if len(active_adapters) == 1 else active_adapters
                )
                vace_wrapped_model.set_adapter(adapter_name)
                logger.info(f"_enable_vace: Restored active adapter(s): {adapter_name}")

        self.components.generator.model = vace_wrapped_model

        self.vace_enabled = True
        logger.info("_enable_vace: VACE enabled successfully")

    def prepare(self, **kwargs) -> Requirements | None:
        """Return input requirements based on current mode."""
        return prepare_for_mode(self.__class__, self.components.config, kwargs)

    def __call__(self, **kwargs) -> torch.Tensor:
        self.first_call, self.last_mode = handle_mode_transition(
            self.state, self.components.vae, self.first_call, self.last_mode, kwargs
        )
        return self._generate(**kwargs)

    def _generate(self, **kwargs) -> torch.Tensor:
        # Lazy load VACE if needed
        if self._needs_vace(kwargs) and not self.vace_enabled:
            self._enable_vace()

        # Handle runtime LoRA scale updates before writing into state.
        lora_scales = kwargs.get("lora_scales")
        if lora_scales is not None:
            self._handle_lora_scale_updates(
                lora_scales=lora_scales, model=self.components.generator.model
            )
            # Trigger cache reset on LoRA scale updates if manage_cache is enabled
            if self.state.get("manage_cache", True):
                kwargs["init_cache"] = True

        # Log vace_ref_images before setting into state
        if "vace_ref_images" in kwargs:
            ref_images = kwargs.get("vace_ref_images", [])
            logger.info(
                f"LongLivePipeline._generate: Setting vace_ref_images into state: "
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

        # Clear video from state if not provided to prevent stale video data
        if "video" not in kwargs:
            self.state.set("video", None)

        # Clear vace_ref_images from state if not provided to prevent encoding on chunks where they weren't sent
        if "vace_ref_images" not in kwargs:
            self.state.set("vace_ref_images", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
