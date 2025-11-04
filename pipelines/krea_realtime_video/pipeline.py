import logging
import time

import torch
from diffusers.modular_pipelines import PipelineState

from lib.schema import Quantization

from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from .blocks.modular_blocks import KreaRealtimeVideoBlocks
from .components.generator import KreaGeneratorWrapper
from .components.scheduler import KreaSchedulerWrapper
from .components.vendor.wan2_1.modules.causal_model import (
    KV_CACHE_ATTENTION_BIAS_DISABLED,
)
from .components.vendor.wan2_1.scheduler import FlowMatchScheduler
from .components.vendor.wan2_1.vae_block3 import WanVAEWrapper
from .components.vendor.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder

logger = logging.getLogger(__name__)

# The VAE compresses a pixel frame into a latent frame which consists of patches
# The patch embedding converts spatial patches into tokens
# The VAE does 8x spatial downsampling
# The patch embedding does 2x spatial downsampling
# Thus, we end up spatially scaling down by 16
SCALE_SIZE = 16

# The VAE does 8x spatial downsampling
VAE_SPATIAL_DOWNSAMPLE_FACTOR = 8

# https://github.com/daydreamlive/scope/blob/a6a7aa1d7a3be60d3b444e254f83a9fd09e9151a/pipelines/base/wan2_1/modules/causal_model.py#L117
MAX_ROPE_FREQ_TABLE_SEQ_LEN = 1024

WARMUP_RUNS = 3
WARMUP_PROMPT = "a majestic sunset"


class ComponentProvider:
    """Simple wrapper to provide component access to modular blocks."""

    def __init__(self, pipeline):
        """
        Initialize the component provider.

        Args:
            pipeline: The pipeline instance that contains all components
        """
        self.pipeline = pipeline

    @property
    def text_encoder(self):
        """Provide access to the text_encoder component."""
        return self.pipeline.text_encoder

    @property
    def vae(self):
        """Provide access to the vae component."""
        return self.pipeline.vae

    @property
    def generator(self):
        """Provide access to the generator component."""
        return self.pipeline.generator

    @property
    def scheduler(self):
        """Provide access to the scheduler component."""
        return self.pipeline.scheduler


class KreaRealtimeVideoPipeline(Pipeline):
    def __init__(
        self,
        config,
        low_memory: bool = False,
        quantization: Quantization | None = None,
        compile: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.seed = getattr(config, "seed", 42)
        self.device = device
        self.dtype = dtype
        self.low_memory = low_memory
        self.quantization = quantization
        self.compile = compile

        # The height and width must be divisible by SCALE_SIZE
        req_height = config.get("height", 512)
        req_width = config.get("width", 512)
        self.height = round(req_height / SCALE_SIZE) * SCALE_SIZE
        self.width = round(req_width / SCALE_SIZE) * SCALE_SIZE

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)

        # Get model_kwargs and extract timestep_shift for scheduler initialization
        model_kwargs = getattr(config, "model_kwargs", {})
        timestep_shift = model_kwargs.get("timestep_shift", 8.0)

        # Create Scheduler
        base_scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler = KreaSchedulerWrapper(
            base_scheduler, num_inference_steps=1000, training=True
        )

        # Create Generator
        model_name = "Wan2.1-T2V-14B"
        self.generator = self._create_generator(
            model_kwargs=model_kwargs,
            model_name=model_name,
            model_dir=model_dir,
            generator_path=generator_path,
            scheduler=self.scheduler,
            quantization=quantization,
            dtype=dtype,
            device=device,
            compile=compile,
        )

        # Create Text Encoder
        start = time.time()
        text_encoder = WanTextEncoder(
            model_name=model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")
        # Move text encoder to target device but use dtype of weights
        self.text_encoder = text_encoder.to(device=device)

        # Create VAE
        start = time.time()
        vae = WanVAEWrapper(
            model_name=model_name, model_dir=model_dir, vae_path=vae_path
        )
        print(f"Loaded VAE in {time.time() - start:.3f}s")
        # Move VAE to target device and use target dtype
        self.vae = vae.to(device=device, dtype=dtype)

        self.stream_denoising_step_list = torch.tensor(
            config.denoising_step_list, dtype=torch.long
        )
        if config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.stream_denoising_step_list = timesteps[
                1000 - self.stream_denoising_step_list
            ]

        num_frame_per_block = config.get("num_frame_per_block", 1)
        print(f"KV inference with {num_frame_per_block} frames per block")

        if num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = num_frame_per_block

        # Configure kv_cache with attributes from config
        frame_seq_length = (self.height // SCALE_SIZE) * (self.width // SCALE_SIZE)
        batch_size = 1
        num_frame_per_block = config.get("num_frame_per_block", 1)
        kv_cache_num_frames = config.get("kv_cache_num_frames", 3)
        local_attn_size = kv_cache_num_frames + num_frame_per_block

        # Configure the kv_cache component
        self.generator.configure(
            frame_seq_length=frame_seq_length,
            batch_size=batch_size,
            num_frame_per_block=num_frame_per_block,
            kv_cache_num_frames=kv_cache_num_frames,
            vae=vae,
        )

        self.local_attn_size = local_attn_size
        self.kv_cache_attention_bias = config.get(
            "kv_cache_attention_bias", KV_CACHE_ATTENTION_BIAS_DISABLED
        )

        self.prompts = None
        self.denoising_step_list = None

        # Initialize Modular Diffusers blocks
        self.modular_blocks = KreaRealtimeVideoBlocks()

        # Create component provider for modular blocks
        self.component_provider = ComponentProvider(self)

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._reset_cache_for_transition
        )

        # Warmup
        # Always warmup regardless of whether compile = True because even if compile = False
        # flex attention will still be compiled
        start = time.time()

        self.prepare(prompts=[{"text": WARMUP_PROMPT}], should_prepare=True)
        for _ in range(WARMUP_RUNS):
            # Use modular blocks for warmup as well
            state = PipelineState()
            if (
                self.generator.conditional_dict is not None
                and "prompt_embeds" in self.generator.conditional_dict
            ):
                state.set(
                    "prompt_embeds", self.generator.conditional_dict["prompt_embeds"]
                )
            if self.stream_denoising_step_list is not None:
                state.set("denoising_step_list", self.stream_denoising_step_list)
            state.set("current_start", self.generator.current_start)
            state.set("base_seed", self.seed)
            state.set("num_frame_per_block", self.generator.num_frame_per_block)
            state.set("height", self.height)
            state.set("width", self.width)
            state.set("init_cache", False)
            _, state = self.modular_blocks(self.component_provider, state)
            # Update current_start for next warmup iteration
            self.generator.current_start += self.generator.num_frame_per_block

        print(f"Warmed up in {time.time() - start:2f}s")

        # Assume that caller will call prepare() to initialize pipeline properly

    def _create_generator(
        self,
        model_kwargs: dict,
        model_name: str,
        model_dir: str | None,
        generator_path: str | None,
        scheduler,
        quantization: Quantization | None,
        dtype: torch.dtype,
        device: torch.device | None,
        compile: bool,
    ):
        start = time.time()

        generator = WanDiffusionWrapper(
            **model_kwargs,
            model_name=model_name,
            model_dir=model_dir,
            is_causal=True,
            generator_path=generator_path,
        )

        # TODO: Unwrap generator to be separated from the scheduler, they should be bound together in the denoise step
        generator.scheduler = scheduler

        print(f"Loaded diffusion wrapper in {time.time() - start:.3f}s")

        for block in generator.model.blocks:
            block.self_attn.fuse_projections()

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

        return KreaGeneratorWrapper(generator)

    def _reset_cache_for_transition(self):
        """Reset cross-attention cache for prompt transitions."""
        generator_param = next(self.generator.model.parameters())
        self.generator.initialize_crossattn_cache(
            batch_size=1, dtype=generator_param.dtype, device=generator_param.device
        )

    def _initialize_cache(self):
        """Initialize KV cache and related buffers."""
        self.generator.initialize_full_cache(
            local_attn_size=self.local_attn_size,
            height=self.height,
            width=self.width,
            low_memory=self.low_memory,
            max_rope_freq_table_seq_len=MAX_ROPE_FREQ_TABLE_SEQ_LEN,
            vae_spatial_downsample_factor=VAE_SPATIAL_DOWNSAMPLE_FACTOR,
        )

    @torch.no_grad()
    def _prepare_stream(
        self,
        prompts: list[str] = None,
        denoising_step_list: list[int] = None,
        init_cache: bool = False,
        kv_cache_attention_bias: float | None = None,
    ):
        if prompts is not None:
            generator_param = next(self.generator.model.parameters())
            # Make sure text encoder is on right device
            self.text_encoder = self.text_encoder.to(generator_param.device)

            # If in low memory mode offload text encoder to CPU
            if self.low_memory:
                self.text_encoder = self.text_encoder.to(torch.device("cpu"))

        if denoising_step_list is not None:
            self.stream_denoising_step_list = torch.tensor(
                denoising_step_list, dtype=torch.long
            )

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        # If caller requested prepare assume cache init
        # Otherwise no cache init
        init_cache = should_prepare

        manage_cache = kwargs.get("manage_cache", None)
        prompts = kwargs.get("prompts", None)
        prompt_interpolation_method = kwargs.get(
            "prompt_interpolation_method", "linear"
        )
        transition = kwargs.get("transition", None)
        denoising_step_list = kwargs.get("denoising_step_list", None)
        kv_cache_attention_bias = kwargs.get("kv_cache_attention_bias", None)

        # Check if prompts changed using prompt blender
        if self.prompt_blender.should_update(prompts, prompt_interpolation_method):
            logger.info("prepare: Initiating pipeline prepare for prompt update")
            should_prepare = True

        # Handle prompt transition requests (with autocast for quantized models)
        with torch.autocast(str(self.device), dtype=self.dtype):
            should_prepare_from_transition, target_prompts = handle_transition_prepare(
                transition, self.prompt_blender, self.text_encoder
            )
        if target_prompts:
            self.prompts = target_prompts
        if should_prepare_from_transition:
            should_prepare = True

        if (
            denoising_step_list is not None
            and denoising_step_list != self.denoising_step_list
        ):
            should_prepare = True

            if manage_cache:
                init_cache = True

        if should_prepare:
            # Update internal state
            if denoising_step_list is not None:
                self.denoising_step_list = denoising_step_list

            # Apply prompt blending and prepare stream
            # (PromptBlender.blend() returns None if transitioning, which skips preparation)
            self._apply_prompt_blending(
                prompts,
                prompt_interpolation_method,
                denoising_step_list,
                init_cache,
                kv_cache_attention_bias,
            )

        return None

    def __call__(
        self,
        _: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        # Update prompt embedding for this generation call
        # Handles both static blending and temporal transitions
        with torch.autocast(str(self.device), dtype=self.dtype):
            next_embedding = self.prompt_blender.get_next_embedding(self.text_encoder)

        if next_embedding is not None:
            # Ensure embedding is in the correct dtype for cross-attention
            next_embedding = next_embedding.to(dtype=self.dtype)
            self.generator.conditional_dict = {"prompt_embeds": next_embedding}

        # Note: The caller must call prepare() before __call__()
        # Use modular blocks instead of direct stream() call
        state = PipelineState()

        # Set up state for modular blocks
        # Set prompt_embeds if available from conditional_dict
        if (
            self.generator.conditional_dict is not None
            and "prompt_embeds" in self.generator.conditional_dict
        ):
            state.set("prompt_embeds", self.generator.conditional_dict["prompt_embeds"])

        # Set denoising step list
        if self.stream_denoising_step_list is not None:
            state.set("denoising_step_list", self.stream_denoising_step_list)

        # Set current_start
        state.set("current_start", self.generator.current_start)

        # Set configuration values
        state.set("base_seed", self.seed)
        state.set("num_frame_per_block", self.generator.num_frame_per_block)
        state.set("height", self.height)
        state.set("width", self.width)
        state.set("init_cache", False)  # Cache already initialized in prepare()

        # Execute modular blocks (returns tuple: components, state)
        _, state = self.modular_blocks(self.component_provider, state)

        # Get output from state
        output = state.values.get("output")

        if output is None:
            raise RuntimeError("Modular blocks did not produce output")

        # Update current_start for next iteration
        self.generator.current_start += self.generator.num_frame_per_block

        # Postprocess output (same as stream would do)
        from ..process import postprocess_chunk

        return postprocess_chunk(output)

    def _apply_prompt_blending(
        self,
        prompts=None,
        interpolation_method="linear",
        denoising_step_list=None,
        init_cache: bool = False,
        kv_cache_attention_bias: float | None = None,
    ):
        """Apply weighted blending of cached prompt embeddings."""
        # autocast to target dtype since we the text encoder weights dtype
        # might be different (eg float8_e4m3fn)
        with torch.autocast(str(self.device), dtype=self.dtype):
            combined_embeds = self.prompt_blender.blend(
                prompts, interpolation_method, self.text_encoder
            )

        if combined_embeds is None:
            return

        # Ensure embedding is in the correct dtype for cross-attention
        combined_embeds = combined_embeds.to(dtype=self.dtype)

        # Set the blended embeddings on kv_cache
        self.generator.conditional_dict = {"prompt_embeds": combined_embeds}

        # Update kv_cache_attention_bias if provided
        if kv_cache_attention_bias is not None:
            self.kv_cache_attention_bias = kv_cache_attention_bias
            logger.info(
                f"prepare: Updated KV cache attention bias to {kv_cache_attention_bias}"
            )

        # Call stream prepare to update the pipeline with denoising steps
        self._prepare_stream(
            prompts=None,
            denoising_step_list=denoising_step_list,
            init_cache=init_cache,
            kv_cache_attention_bias=kv_cache_attention_bias,
        )

        # Initialize cache if needed
        if init_cache:
            self._initialize_cache()
