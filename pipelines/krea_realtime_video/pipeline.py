import logging
import time

import torch

from lib.schema import Quantization

from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from .inference import InferencePipeline
from .vendor.wan2_1.vae_block3 import WanVAEWrapper
from .vendor.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder

logger = logging.getLogger(__name__)


class KreaRealtimeVideoPipeline(Pipeline):
    def __init__(
        self,
        config,
        low_memory: bool = False,
        quantization: Quantization | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)
        vae_path = getattr(config, "vae_path", None)

        # Load diffusion model
        start = time.time()
        model_name = "Wan2.1-T2V-14B"
        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}),
            model_name=model_name,
            model_dir=model_dir,
            is_causal=True,
            generator_path=generator_path,
        )

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

        start = time.time()
        text_encoder = WanTextEncoder(
            model_name=model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")

        start = time.time()
        vae = WanVAEWrapper(
            model_name=model_name, model_dir=model_dir, vae_path=vae_path
        )
        print(f"Loaded VAE in {time.time() - start:.3f}s")

        seed = getattr(config, "seed", 42)

        # Move text encoder to target device but use dtype of weights
        text_encoder = text_encoder.to(device=device)
        # Move VAE to target device and use target dtype
        vae = vae.to(device=device, dtype=dtype)

        self.stream = InferencePipeline(
            config, generator, text_encoder, vae, low_memory, seed
        )
        self.device = device
        self.dtype = dtype

        self.prompts = None
        self.denoising_step_list = None

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._reset_cache_for_transition
        )

    def _reset_cache_for_transition(self):
        """Reset cross-attention cache for prompt transitions."""
        generator_param = next(self.stream.generator.model.parameters())
        self.stream._initialize_crossattn_cache(
            batch_size=1, dtype=generator_param.dtype, device=generator_param.device
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

        # Check if prompts changed using prompt blender
        if self.prompt_blender.should_update(prompts, prompt_interpolation_method):
            logger.info("prepare: Initiating pipeline prepare for prompt update")
            should_prepare = True

        # Handle prompt transition requests (with autocast for quantized models)
        with torch.autocast(str(self.device), dtype=self.dtype):
            should_prepare_from_transition, target_prompts = handle_transition_prepare(
                transition, self.prompt_blender, self.stream.text_encoder
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
                prompts, prompt_interpolation_method, denoising_step_list, init_cache
            )

        return None

    def __call__(
        self,
        _: torch.Tensor | list[torch.Tensor] | None = None,
    ):
        # Update prompt embedding for this generation call
        # Handles both static blending and temporal transitions
        with torch.autocast(str(self.device), dtype=self.dtype):
            next_embedding = self.prompt_blender.get_next_embedding(
                self.stream.text_encoder
            )

        if next_embedding is not None:
            # Ensure embedding is in the correct dtype for cross-attention
            next_embedding = next_embedding.to(dtype=self.dtype)
            self.stream.conditional_dict = {"prompt_embeds": next_embedding}

        # Note: The caller must call prepare() before __call__()
        return self.stream()

    def _apply_prompt_blending(
        self,
        prompts=None,
        interpolation_method="linear",
        denoising_step_list=None,
        init_cache: bool = False,
    ):
        """Apply weighted blending of cached prompt embeddings."""
        # autocast to target dtype since we the text encoder weights dtype
        # might be different (eg float8_e4m3fn)
        with torch.autocast(str(self.device), dtype=self.dtype):
            combined_embeds = self.prompt_blender.blend(
                prompts, interpolation_method, self.stream.text_encoder
            )

        if combined_embeds is None:
            return

        # Ensure embedding is in the correct dtype for cross-attention
        combined_embeds = combined_embeds.to(dtype=self.dtype)

        # Set the blended embeddings on the stream
        self.stream.conditional_dict = {"prompt_embeds": combined_embeds}

        # Call stream prepare to update the pipeline with denoising steps
        self.stream.prepare(
            prompts=None, denoising_step_list=denoising_step_list, init_cache=init_cache
        )
