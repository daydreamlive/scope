import json
import logging
import os
import time
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline
from ..process import postprocess_chunk
from ..utils import load_model_config, load_state_dict, validate_resolution
from ..wan2_1.components.text_encoder import WanTextEncoderWrapper
from ..wan2_1.vae import create_vae
from .modules.sampling import rcm_sample
from .modules.sla_attention import replace_attention_with_sla
from .schema import TurboDiffusionConfig

if TYPE_CHECKING:
    from ..base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class TurboDiffusionPipeline(Pipeline):
    """TurboDiffusion pipeline for accelerated Wan2.1 video generation.

    Uses rCM (Rectified Consistency Model) timestep distillation for 1-4 step
    generation and optional SLA (Sparse-Linear Attention) for additional speedup.
    Generates complete videos in one shot (non-streaming/non-autoregressive).
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TurboDiffusionConfig

    def __init__(
        self,
        config,
        quantization=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from ..longlive.modules.model import WanModel

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype

        # Store generation parameters
        self.height = config.height
        self.width = config.width
        self.num_frames = getattr(config, "num_frames", 81)
        self.num_steps = getattr(config, "num_steps", 4)
        self.sigma_max = getattr(config, "sigma_max", 80.0)
        self.base_seed = getattr(config, "base_seed", 42)

        # Validate resolution (VAE downsample 8 * patch embedding downsample 2 = 16)
        validate_resolution(
            height=self.height,
            width=self.width,
            scale_factor=16,
        )

        model_dir = getattr(config, "model_dir", None)
        generator_path = getattr(config, "generator_path", None)
        text_encoder_path = getattr(config, "text_encoder_path", None)
        tokenizer_path = getattr(config, "tokenizer_path", None)

        model_config = load_model_config(config, __file__)
        base_model_name = getattr(model_config, "base_model_name", "Wan2.1-T2V-1.3B")

        # --- Load diffusion model (WanModel) ---
        start = time.time()

        model_dir = model_dir if model_dir is not None else "wan_models"
        model_path = os.path.join(model_dir, f"{base_model_name}/")
        config_path = os.path.join(model_path, "config.json")

        with open(config_path) as f:
            model_cfg = json.load(f)

        # Load rCM-distilled checkpoint
        state_dict = load_state_dict(generator_path)

        # Remove 'model.' prefix if present (from wrapped models)
        if state_dict and all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        # Filter config to WanModel's accepted params
        import inspect

        sig = inspect.signature(WanModel.__init__)
        filtered_cfg = {k: v for k, v in model_cfg.items() if k in sig.parameters}

        with torch.device("meta"):
            self.generator = WanModel(**filtered_cfg)

        # Materialize on CPU, then load weights
        self.generator = self.generator.to_empty(device="cpu")
        self.generator.load_state_dict(state_dict, assign=True, strict=False)

        # Reinitialize freqs (not in state_dict)
        if hasattr(self.generator, "freqs"):
            from ..longlive.modules.model import rope_params

            d = self.generator.dim // self.generator.num_heads
            self.generator.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )

        self.generator.eval()
        self.generator.requires_grad_(False)
        del state_dict

        # Compute seq_len for positional encoding
        self.seq_len = 32760  # default for global attention

        print(f"Loaded TurboDiffusion model in {time.time() - start:.3f}s")

        # --- Apply SLA attention replacement (before moving to device) ---
        attention_type = getattr(config, "attention_type", "sagesla")
        sla_topk = getattr(config, "sla_topk", 0.12)
        if attention_type != "original":
            replace_attention_with_sla(
                self.generator, attention_type=attention_type, topk=sla_topk
            )

        # --- Quantization ---
        from ..enums import Quantization

        if quantization == Quantization.FP8_E4M3FN:
            self.generator = self.generator.to(dtype=dtype)
            start = time.time()

            from torchao.quantization.quant_api import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                quantize_,
            )

            quantize_(
                self.generator,
                Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
                device=self.device,
            )
            print(
                f"Quantized TurboDiffusion model to fp8 in {time.time() - start:.3f}s"
            )
        else:
            self.generator = self.generator.to(device=self.device, dtype=dtype)

        # --- Load text encoder ---
        start = time.time()
        self.text_encoder = WanTextEncoderWrapper(
            model_name=base_model_name,
            model_dir=model_dir,
            text_encoder_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        print(f"Loaded text encoder in {time.time() - start:.3f}s")
        self.text_encoder = self.text_encoder.to(device=self.device)

        # --- Load VAE ---
        vae_type = getattr(config, "vae_type", "wan")
        start = time.time()
        self.vae = create_vae(
            model_dir=model_dir, model_name=base_model_name, vae_type=vae_type
        )
        print(f"Loaded VAE (type={vae_type}) in {time.time() - start:.3f}s")
        self.vae = self.vae.to(device=self.device, dtype=dtype)

    def prepare(self, **kwargs):
        """No input frames needed — text-only batch generation."""
        return None

    def __call__(self, **kwargs) -> dict:
        prompt = kwargs.get("prompt", "")
        num_steps = kwargs.get("num_steps", self.num_steps)
        sigma_max = kwargs.get("sigma_max", self.sigma_max)
        base_seed = kwargs.get("base_seed", self.base_seed)

        # 1. Text encode
        if isinstance(prompt, str):
            prompt = [prompt]
        text_output = self.text_encoder(prompt)
        prompt_embeds = text_output["prompt_embeds"]

        # Convert to list of [L, C] tensors for WanModel
        context = [prompt_embeds[i] for i in range(prompt_embeds.shape[0])]

        # 2. Compute latent shape
        # VAE spatial compression: 8x, temporal compression: 4x
        latent_channels = 16  # Wan2.1 latent channels
        latent_frames = (self.num_frames - 1) // 4 + 1  # temporal compression
        latent_h = self.height // 8
        latent_w = self.width // 8
        latent_shape = (latent_channels, latent_frames, latent_h, latent_w)

        # 3. Setup generator for reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(base_seed)

        # 4. rCM sampling (1-4 steps)
        samples = rcm_sample(
            model=self.generator,
            context=context,
            seq_len=self.seq_len,
            latent_shape=latent_shape,
            num_steps=num_steps,
            sigma_max=sigma_max,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        # 5. VAE decode
        # samples shape: [B, C, T, H, W] -> VAE expects [B, T, C, H, W]
        samples = samples.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            video = self.vae.decode_to_pixel(samples, use_cache=False)

        # video shape: [B, T, C, H, W] in range [-1, 1]
        # postprocess_chunk expects [B, T, C, H, W] and outputs THWC [0, 1]
        return {"video": postprocess_chunk(video)}
