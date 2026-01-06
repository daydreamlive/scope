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
from ..wan2_1.vae import create_vae
from .modular_blocks import KreaRealtimeVideoBlocks
from .schema import KreaRealtimeVideoConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]
# This default value < 1.0 will trigger torch.compile in the flex_attention code path
# during warmup
DEFAULT_KV_CACHE_ATTENTION_BIAS = 0.3

WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline):
    # VACE state - lazy loaded when needed
    vace_enabled: bool = False
    _vace_path: str | None = None
    _vace_in_dim: int = 96
    _vace_layers: list[int] | None = None
    _stored_device: torch.device | None = None
    _stored_dtype: torch.dtype | None = None
    _stored_model_dir: str | None = None

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return KreaRealtimeVideoConfig

    def __init__(
        self,
        config,
        quantization: Quantization | None = None,
        compile: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from .modules.causal_model import CausalWanModel

        # Validate resolution requirements
        # VAE downsample (8) * patch embedding downsample (2) = 16
        validate_resolution(
            height=config.height,
            width=config.width,
            scale_factor=16,
        )

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)

        model_config = load_model_config(config, __file__)

        # Store VACE path and config for lazy loading
        self._vace_path = getattr(config, "vace_path", None)
        self._vace_in_dim = getattr(model_config, "vace_in_dim", 96)
        self._vace_layers = getattr(model_config, "vace_layers", None)
        self._stored_device = device
        self._stored_dtype = dtype
        self._stored_model_dir = model_dir
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

        # Load VAE using create_vae factory (supports multiple VAE types)
        vae_type = getattr(config, "vae_type", "wan")
        start = time.time()
        vae = create_vae(
            model_dir=model_dir,
            model_name=base_model_name,
            vae_type=vae_type,
            vae_path=vae_path,
        )
        print(f"Loaded VAE (type={vae_type}) in {time.time() - start:.3f}s")
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
        self.last_mode = None  # Track mode for transition detection

    def _needs_vace(self, kwargs: dict) -> bool:
        """Check if VACE is needed based on the provided kwargs."""
        if self._vace_path is None:
            return False
        # Check for VACE-specific inputs or explicit vace_enabled parameter
        return (
            kwargs.get("vace_enabled") is True
            or kwargs.get("vace_ref_images") is not None
            or kwargs.get("vace_input_frames") is not None
        )

    def _enable_vace(self) -> None:
        """Lazy-load VACE components when first needed."""
        if self.vace_enabled:
            return

        if self._vace_path is None:
            logger.warning("_enable_vace: No vace_path configured, cannot enable VACE")
            return

        logger.info("_enable_vace: Lazy loading VACE components...")

        from ..wan2_1.vace import load_vace_weights_only

        # Free up GPU memory by offloading text encoder to CPU
        # The text encoder (~6.4 GB FP8) is only used occasionally for prompt encoding
        # and can run on CPU while outputting embeddings to GPU
        # NOTE: Must cast to BF16 first since FP8 doesn't work on CPU
        logger.info("_enable_vace: Offloading text encoder to CPU to free VRAM...")
        self.components.text_encoder.output_device = (
            self._stored_device
        )  # Keep outputs on GPU
        # Cast to bfloat16 before moving to CPU (FP8 is GPU-only)
        self.components.text_encoder.to(dtype=torch.bfloat16, device="cpu")
        torch.cuda.empty_cache()
        logger.info(
            "_enable_vace: Text encoder offloaded to CPU (as bf16), CUDA cache cleared"
        )

        # Get the current generator model
        base_model = self.components.generator.model

        # Wrap with VACE using stored config (vace_in_dim, vace_layers)
        start = time.time()
        logger.info(
            f"_enable_vace: Wrapping model with CausalVaceWanModel "
            f"(vace_in_dim={self._vace_in_dim}, vace_layers={self._vace_layers})..."
        )
        from ..wan2_1.vace import CausalVaceWanModel

        vace_wrapped_model = CausalVaceWanModel(
            base_model,
            vace_in_dim=self._vace_in_dim,
            vace_layers=self._vace_layers,
        )
        logger.info(f"_enable_vace: Wrapped model in {time.time() - start:.3f}s")

        # Load VACE weights (BF16) on CPU first
        start = time.time()
        logger.info(f"_enable_vace: Loading VACE weights from {self._vace_path}...")
        load_vace_weights_only(vace_wrapped_model, self._vace_path)
        logger.info(f"_enable_vace: Loaded VACE weights in {time.time() - start:.3f}s")

        # Move VACE components to GPU and quantize to FP8 (same as main model)
        # This provides memory efficiency while keeping VACE on GPU for fast inference
        logger.info(
            "_enable_vace: Moving VACE components to GPU and quantizing to FP8..."
        )
        vace_wrapped_model.vace_patch_embedding.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        vace_wrapped_model.vace_blocks.to(
            device=self._stored_device, dtype=self._stored_dtype
        )

        # Quantize VACE components to FP8 (same as main model quantization)
        start = time.time()
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            PerTensor,
            quantize_,
        )

        quantize_(
            vace_wrapped_model.vace_patch_embedding,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            device=self._stored_device,
        )
        quantize_(
            vace_wrapped_model.vace_blocks,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            device=self._stored_device,
        )
        logger.info(
            f"_enable_vace: Quantized VACE to FP8 in {time.time() - start:.3f}s"
        )

        # Replace generator model
        logger.info("_enable_vace: Setting wrapped model as generator...")
        self.components.generator.model = vace_wrapped_model
        logger.info("_enable_vace: Wrapped model set successfully")

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
        if "vace_input_frames" not in kwargs:
            self.state.set("vace_input_frames", None)
        if "vace_input_masks" not in kwargs:
            self.state.set("vace_input_masks", None)

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
