import logging
import time

import torch

from ..base.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from ..blending import PromptBlender, handle_transition_prepare
from ..interface import Pipeline, Requirements
from .inference import InferencePipeline
from .utils.lora_utils import configure_lora_for_model, load_lora_checkpoint

logger = logging.getLogger(__name__)


class LongLivePipeline(Pipeline):
    def __init__(
        self,
        config,
        low_memory: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        lora_path = getattr(config, "lora_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)

        # Load diffusion model
        start = time.time()
        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), model_dir=model_dir, is_causal=True
        )
        print(f"Loaded diffusion wrapper in {time.time() - start:.3f}s")
        # Load state dict for LongLive model
        start = time.time()
        generator_state_dict = torch.load(
            generator_path,
            map_location="cpu",
            mmap=True,
        )
        generator.load_state_dict(generator_state_dict["generator"])
        print(f"Loaded diffusion state dict in {time.time() - start:.3f}s")
        # Configure LoRA for LongLive model
        start = time.time()
        generator.model = configure_lora_for_model(
            generator.model,
            model_name="generator",
            lora_config=config.adapter,
        )
        # Load LoRA weights
        load_lora_checkpoint(generator.model, lora_path)
        print(f"Loaded diffusion LoRA in {time.time() - start:.3f}s")

        start = time.time()
        text_encoder = WanTextEncoder(
            model_dir=model_dir, text_encoder_path=text_encoder_path
        )
        print(f"Loaded text encoder in {time.time() - start:3f}s")

        start = time.time()
        vae = WanVAEWrapper(model_dir=model_dir)
        print(f"Loaded VAE in {time.time() - start:.3f}s")

        seed = getattr(config, "seed", 42)

        self.stream = InferencePipeline(
            config, generator, text_encoder, vae, low_memory, seed
        ).to(device=device, dtype=dtype)

        self.prompts = None
        self.denoising_step_list = None

        # Prompt blending with cache reset callback for transitions
        self.prompt_blender = PromptBlender(
            device, dtype, cache_reset_callback=self._reset_cache_for_transition
        )

    def _reset_cache_for_transition(self):
        """Reset cross-attention cache for prompt transitions."""
        # Use model's current device/dtype (should match initialization)
        model_param = next(self.stream.generator.model.parameters())
        self.stream._initialize_crossattn_cache(
            batch_size=1, dtype=model_param.dtype, device=model_param.device
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

        # Handle prompt transition requests
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
        next_embedding = self.prompt_blender.get_next_embedding(
            self.stream.text_encoder
        )

        if next_embedding is not None:
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

        combined_embeds = self.prompt_blender.blend(
            prompts, interpolation_method, self.stream.text_encoder
        )

        if combined_embeds is None:
            return

        # Set the blended embeddings on the stream
        self.stream.conditional_dict = {"prompt_embeds": combined_embeds}

        # Call stream prepare to update the pipeline with denoising steps
        self.stream.prepare(
            prompts=None, denoising_step_list=denoising_step_list, init_cache=init_cache
        )
