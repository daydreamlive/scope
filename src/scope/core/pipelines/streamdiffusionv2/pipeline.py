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
from ..schema import StreamDiffusionV2Config
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.vace import VACEEnabledPipeline
from .components import StreamDiffusionV2WanVAEWrapper
from .modular_blocks import StreamDiffusionV2Blocks
from .modules.causal_model import CausalWanModel

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [750, 250]


class StreamDiffusionV2Pipeline(Pipeline, LoRAEnabledPipeline, VACEEnabledPipeline):
    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return StreamDiffusionV2Config

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
        vace_path = getattr(config, "vace_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")
        base_model_kwargs = getattr(model_config, "base_model_kwargs", {})
        generator_model_name = getattr(
            model_config, "generator_model_name", "generator"
        )

        # Load generator with VACE support via upfront loading
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

        # Load VAE using unified WanVAEWrapper
        start = time.time()
        vae = StreamDiffusionV2WanVAEWrapper(
            model_dir=model_dir, model_name=base_model_name
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

        This is called lazily on first use when VACE is needed.

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

        # Wrap model with VACE
        self.components.generator.model = CausalVaceWanModel(
            self.components.generator.model, vace_in_dim=self.vace_in_dim
        )

        # Move VACE-specific components to correct device/dtype
        # The wrapped model's VACE components (vace_patch_embedding, vace_blocks) were created
        # on CPU with default dtype. We need to move them to match the base model.
        self.components.generator.model.vace_patch_embedding.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        self.components.generator.model.vace_blocks.to(
            device=self._stored_device, dtype=self._stored_dtype
        )

        # Load VACE weights
        from ..wan2_1.vace import load_vace_weights_only

        load_vace_weights_only(self.components.generator.model, self.vace_path)

        self.vace_enabled = True

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
