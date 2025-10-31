from .wan.wan_wrapper import (
    WanTextEncoder,
    WanVAEWrapper,
    WanDiffusionWrapper,
    CausalWanDiffusionWrapper,
)
from .sdxl.sdxl_wrapper import SDXLWrapper, SDXLTextEncoder, SDXLVAE
from transformers.models.t5.modeling_t5 import T5Block


DIFFUSION_NAME_TO_CLASS = {
    "sdxl": SDXLWrapper,
    "wan": WanDiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper,
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "sdxl": SDXLTextEncoder,
    "wan": WanTextEncoder,
    "causal_wan": WanTextEncoder,
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "sdxl": SDXLVAE,
    "wan": WanVAEWrapper,
    "causal_wan": WanVAEWrapper,
}


def get_vae_wrapper(model_name):
    """
    get_vae_wrapper: Get VAE wrapper class for the given model name.

    Priority order:
    1. Vendor-specific implementations (wan, causal_wan, sdxl) - optimized for this pipeline
    2. Factory implementations (wan2.1, lightvae2.1) - shared across pipelines

    This ensures vendor optimizations are preserved while allowing factory VAEs.
    """
    # Check vendor registry first (preserves vendor-specific optimizations)
    if model_name in VAE_NAME_TO_CLASS:
        return VAE_NAME_TO_CLASS[model_name]

    # Fall back to factory registry for shared VAE implementations
    try:
        from pipelines.base.wan2_1.vae_factory import VAE_CONFIGS, create_vae_wrapper
        from pipelines.base.wan2_1.stream_vae_adapter import StreamVAEAdapter

        if model_name in VAE_CONFIGS:
            def create_adapted_vae(model_dir=None):
                base_vae = create_vae_wrapper(vae_model=model_name, model_dir=model_dir)
                return StreamVAEAdapter(base_vae)
            return create_adapted_vae
    except ImportError:
        pass

    # Model name not found in either registry
    available_vendor = ", ".join(f"'{k}'" for k in VAE_NAME_TO_CLASS.keys())
    try:
        from ....base.wan2_1.vae_factory import VAE_CONFIGS
        available_factory = ", ".join(f"'{k}'" for k in VAE_CONFIGS.keys())
        available = f"{available_vendor}, {available_factory}"
    except ImportError:
        available = available_vendor

    raise ValueError(
        f"get_vae_wrapper: Unknown VAE model '{model_name}'. "
        f"Available models: {available}"
    )


BLOCK_NAME_TO_BLOCK_CLASS = {"T5Block": T5Block}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]
