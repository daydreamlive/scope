from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, SettingsControlType
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_14B_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..utils import VaeType


class KreaRealtimeVideoConfig(BasePipelineConfig):
    pipeline_id = "krea-realtime-video"
    pipeline_name = "Krea Realtime Video"
    pipeline_description = (
        "A streaming pipeline and autoregressive video diffusion model from Krea. "
        "The model is trained using Self-Forcing on Wan2.1 14b."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/krea_realtime_video/docs/usage.md"
    estimated_vram_gb = 32.0
    supports_lora = True
    supports_vace = True
    artifacts = [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_14B_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            files=["config.json"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="krea/krea-realtime-video",
            files=["krea-realtime-video-14b.safetensors"],
        ),
    ]

    min_dimension = 16
    modified = True
    supports_quantization = True
    recommended_quantization_vram_threshold = 40.0

    default_temporal_interpolation_method = "linear"
    default_temporal_interpolation_steps = 4

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
                "base_seed",
                SettingsControlType.CACHE_MANAGEMENT,
                "kv_cache_attention_bias",
                SettingsControlType.DENOISING_STEPS,
                "quantization",
            ],
        ),
        "video": ModeDefaults(
            height=256,
            width=256,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
            default_temporal_interpolation_steps=0,
            # Video mode includes noise controls
            settings_panel=[
                SettingsControlType.VACE,
                SettingsControlType.LORA,
                SettingsControlType.PREPROCESSOR,
                "vae_type",
                "height",
                "width",
                "base_seed",
                SettingsControlType.CACHE_MANAGEMENT,
                "kv_cache_attention_bias",
                SettingsControlType.DENOISING_STEPS,
                SettingsControlType.NOISE_CONTROLS,
                "quantization",
            ],
        ),
    }
