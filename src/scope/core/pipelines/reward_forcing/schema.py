from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, SettingsControlType
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..utils import VaeType


class RewardForcingConfig(BasePipelineConfig):
    pipeline_id = "reward-forcing"
    pipeline_name = "RewardForcing"
    pipeline_description = (
        "A streaming pipeline and autoregressive video diffusion model from ZJU, Ant Group, SIAS-ZJU, HUST and SJTU. "
        "The model is trained with Rewarded Distribution Matching Distillation using Wan2.1 1.3b as the base model."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/reward_forcing/docs/usage.md"
    estimated_vram_gb = 20.0
    supports_lora = True
    supports_vace = True
    artifacts = [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="JaydenLu666/Reward-Forcing-T2V-1.3B",
            files=["rewardforcing.pt"],
        ),
    ]

    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16
    modified = True

    height: int = 320
    width: int = 576
    denoising_steps: list[int] = [1000, 750, 500, 250]
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use for encoding/decoding. 'wan' is the full VAE with best quality. 'lightvae' is 75% pruned for faster performance but lower quality. 'tae' is a tiny autoencoder for fast preview quality. 'lighttae' is LightTAE with WanVAE normalization for faster performance with consistent latent space.",
    )

    modes = {
        "text": ModeDefaults(
            default=True,
            # Settings panel for text mode (no noise controls)
            settings_panel=[
                SettingsControlType.VACE,
                SettingsControlType.LORA,
                SettingsControlType.PREPROCESSOR,
                "vae_type",
                "height",
                "width",
                "seed",
                SettingsControlType.CACHE_MANAGEMENT,
                SettingsControlType.DENOISING_STEPS,
                "quantization",
            ],
        ),
        "video": ModeDefaults(
            height=512,
            width=512,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
            # Video mode includes noise controls
            settings_panel=[
                SettingsControlType.VACE,
                SettingsControlType.LORA,
                SettingsControlType.PREPROCESSOR,
                "vae_type",
                "height",
                "width",
                "seed",
                SettingsControlType.CACHE_MANAGEMENT,
                SettingsControlType.DENOISING_STEPS,
                SettingsControlType.NOISE_CONTROLS,
                "quantization",
            ],
        ),
    }
