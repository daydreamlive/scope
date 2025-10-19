import logging
import time

import torch

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
        use_fp8: bool = False,
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

        if use_fp8:
            start = time.time()

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            quantize_(
                generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
            )

            print(f"Quantized diffusion model to fp8 in {time.time() - start:.3f}s")

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

        self.stream = InferencePipeline(
            config, generator, text_encoder, vae, low_memory, seed
        ).to(device=device, dtype=dtype)

        self.prompts = None
        self.denoising_step_list = None

    def prepare(self, should_prepare: bool = False, **kwargs) -> Requirements | None:
        # If caller requested prepare assume cache init
        # Otherwise no cache init
        init_cache = should_prepare

        manage_cache = kwargs.get("manage_cache", None)
        prompts = kwargs.get("prompts", None)
        denoising_step_list = kwargs.get("denoising_step_list", None)

        if prompts is not None and prompts != self.prompts:
            should_prepare = True

        if (
            denoising_step_list is not None
            and denoising_step_list != self.denoising_step_list
        ):
            should_prepare = True

            if manage_cache:
                init_cache = True

        if should_prepare:
            if prompts is not None:
                self.prompts = prompts

            if denoising_step_list is not None:
                self.denoising_step_list = denoising_step_list

            self.stream.prepare(
                prompts=self.prompts,
                denoising_step_list=self.denoising_step_list,
                init_cache=init_cache,
            )

        return None

    def __call__(
        self,
        _: torch.Tensor | list[torch.Tensor] | None = None,
        prompts: list[str] = None,
        denoising_step_list: list[int] = None,
        manage_cache: bool = True,
    ):
        self.prepare(
            prompts=prompts,
            denoising_step_list=denoising_step_list,
            manage_cache=manage_cache,
        )
        return self.stream()
