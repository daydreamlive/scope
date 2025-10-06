import time

import torch

from ..base.wan2_1.wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from ..interface import Pipeline, Requirements
from .inference import InferencePipeline
from .utils.lora_utils import configure_lora_for_model, load_lora_checkpoint


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

    def prepare(
        self, prompts: list[str] = None, should_prepare: bool = False
    ) -> Requirements | None:
        # If caller requested prepare assume cache init
        # Otherwise no cache init
        init_cache = should_prepare

        if prompts is not None and prompts != self.prompts:
            self.prompts = prompts
            should_prepare = True

        if should_prepare:
            self.stream.prepare(self.prompts, init_cache=init_cache)

        return None

    def __call__(self, _: torch.Tensor | list[torch.Tensor] | None = None, **kwargs):
        self.stream.prepare()
        return self.stream()
