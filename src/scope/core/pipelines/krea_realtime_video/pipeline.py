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
from ..schema import KreaRealtimeVideoConfig
from ..utils import Quantization, load_model_config
from ..wan2_1.components import WanDiffusionWrapper, WanTextEncoderWrapper
from ..wan2_1.lora.mixin import LoRAEnabledPipeline
from ..wan2_1.vace import CausalVaceWanModel
from ..wan2_1.vae import WanVAEWrapper
from .modular_blocks import KreaRealtimeVideoBlocks
from .modules.causal_model import CausalWanModel

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

DEFAULT_DENOISING_STEP_LIST = [1000, 750, 500, 250]
# This default value < 1.0 will trigger torch.compile in the flex_attention code path
# during warmup
DEFAULT_KV_CACHE_ATTENTION_BIAS = 0.3

WARMUP_PROMPT = [{"text": "a majestic sunset", "weight": 1.0}]


class KreaRealtimeVideoPipeline(Pipeline, LoRAEnabledPipeline):
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
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)
        vace_path = getattr(config, "vace_path", None)

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
        self._stored_quantization = quantization

    def _needs_vace(self, kwargs: dict) -> bool:
        """_needs_vace: Check if VACE is needed for this generation call.

        VACE is needed when:
        - ref_images is present and non-empty, OR
        - input_frames is present (for depth/flow/pose conditioning)

        Args:
            kwargs: Generation call kwargs

        Returns:
            bool: True if VACE is needed
        """
        ref_images = kwargs.get("ref_images")
        input_frames = kwargs.get("input_frames")

        has_ref_images = ref_images is not None and len(ref_images) > 0
        has_input_frames = input_frames is not None
        return has_ref_images or has_input_frames

    def _enable_vace(self):
        """_enable_vace: Enable VACE by wrapping model and loading weights.

        This is called lazily when VACE is first needed (ref_images or input_frames provided).

        Architecture: Since Krea uses CausalWanModel, we wrap it with CausalVaceWanModel
        and load VACE weights. For Krea, this produces: CausalVaceWanModel(CausalWanModel)

        Note: Unlike LongLive, Krea does not have a built-in performance LoRA, so we don't
        need complex PEFT unwrapping/rewrapping logic. We simply wrap the base model with VACE.

        In addition, loads separate VACE VAE for encoding.

        If the base model was quantized, VACE blocks will also be quantized to maintain
        memory efficiency.

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

        # Get base model
        logger.info("_enable_vace: Getting base model...")
        start = time.time()
        base_model = self.components.generator.model
        logger.info(f"_enable_vace: Got base model in {time.time() - start:.3f}s")

        # Wrap with VACE and load weights
        logger.info("_enable_vace: Creating VACE wrapper...")
        start = time.time()

        # Krea uses 14B VACE checkpoint which has 8 VACE blocks
        # For 32-layer model: [0, 4, 8, 12, 16, 20, 24, 28]
        num_layers = base_model.num_layers
        num_vace_blocks = 8
        vace_layers = [
            i * num_layers // num_vace_blocks for i in range(num_vace_blocks)
        ]
        logger.info(
            f"_enable_vace: Using 14B VACE configuration with {len(vace_layers)} blocks at layers {vace_layers}"
        )

        vace_wrapped_model = CausalVaceWanModel(
            base_model, vace_in_dim=self.vace_in_dim, vace_layers=vace_layers
        )
        logger.info(f"_enable_vace: Created VACE wrapper in {time.time() - start:.3f}s")

        logger.info("_enable_vace: Moving vace_patch_embedding to device...")
        start = time.time()
        vace_wrapped_model.vace_patch_embedding.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        logger.info(
            f"_enable_vace: Moved vace_patch_embedding in {time.time() - start:.3f}s"
        )

        logger.info("_enable_vace: Moving vace_blocks to device...")
        start = time.time()
        vace_wrapped_model.vace_blocks.to(
            device=self._stored_device, dtype=self._stored_dtype
        )
        logger.info(f"_enable_vace: Moved vace_blocks in {time.time() - start:.3f}s")

        start = time.time()
        logger.info(f"_enable_vace: Loading VACE weights from {self.vace_path}")
        from ..wan2_1.vace import load_vace_weights_only

        load_vace_weights_only(vace_wrapped_model, self.vace_path)
        logger.info(f"_enable_vace: Loaded VACE weights in {time.time() - start:.3f}s")

        # Apply quantization to VACE blocks if base model was quantized
        if self._stored_quantization == Quantization.FP8_E4M3FN:
            start = time.time()
            logger.info("_enable_vace: Quantizing VACE blocks to fp8...")

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            # Quantize VACE-specific components
            # Note: base_model inside vace_wrapped_model is already quantized
            logger.info("_enable_vace: Quantizing vace_patch_embedding...")
            quantize_(
                vace_wrapped_model.vace_patch_embedding,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=self._stored_device,
            )
            logger.info("_enable_vace: Quantizing vace_blocks...")
            quantize_(
                vace_wrapped_model.vace_blocks,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=self._stored_device,
            )
            logger.info(
                f"_enable_vace: Quantized VACE blocks to fp8 in {time.time() - start:.3f}s"
            )

        logger.info("_enable_vace: Setting wrapped model as generator...")
        self.components.generator.model = vace_wrapped_model
        logger.info("_enable_vace: Wrapped model set successfully")

        # Load separate VACE VAE for encoding
        start = time.time()
        logger.info("_enable_vace: Loading VACE VAE...")
        vace_vae = WanVAEWrapper(
            model_dir=self._stored_model_dir, model_name="Wan2.1-VACE-14B"
        )
        logger.info("_enable_vace: Moving VACE VAE to device...")
        vace_vae = vace_vae.to(device=self._stored_device, dtype=self._stored_dtype)
        self.components.add("vace_vae", vace_vae)
        logger.info(f"_enable_vace: Loaded VACE VAE in {time.time() - start:.3f}s")

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

        if self.state.get("denoising_step_list") is None:
            self.state.set("denoising_step_list", DEFAULT_DENOISING_STEP_LIST)

        # Apply mode-specific defaults (noise_scale, noise_controller)
        mode = resolve_input_mode(kwargs)
        apply_mode_defaults_to_state(self.state, self.__class__, mode, kwargs)

        _, self.state = self.blocks(self.components, self.state)
        return postprocess_chunk(self.state.values["output_video"])
