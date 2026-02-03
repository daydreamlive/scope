from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_14B_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..enums import Quantization
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

    supports_cache_management = True
    supports_kv_cache_bias = True
    supports_quantization = True
    min_dimension = 16
    modified = True
    recommended_quantization_vram_threshold = 40.0

    default_temporal_interpolation_method = "linear"
    default_temporal_interpolation_steps = 4

    vace_context_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Scaling factor for VACE hint injection (0.0 to 2.0)",
        json_schema_extra=ui_field_config(
            order=1, component="vace", is_load_param=True
        ),
    )
    lora_merge_strategy: str = Field(
        default="permanent_merge",
        description="LoRA merge strategy",
        json_schema_extra=ui_field_config(
            order=2, component="lora", is_load_param=True
        ),
    )
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="VAE"),
    )
    height: int = Field(
        default=320,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=576,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=4, component="resolution", is_load_param=True
        ),
    )
    base_seed: int = Field(
        default=42,
        ge=0,
        description="Base random seed for reproducible generation",
        json_schema_extra=ui_field_config(order=5, is_load_param=True, label="Seed"),
    )
    manage_cache: bool = Field(
        default=True,
        description="Enable automatic cache management for performance optimization",
        json_schema_extra=ui_field_config(
            order=5, component="cache", is_load_param=True
        ),
    )
    denoising_steps: list[int] = Field(
        default=[1000, 750, 500, 250],
        description="Denoising step schedule for progressive generation",
        json_schema_extra=ui_field_config(
            order=6, component="denoising_steps", is_load_param=True
        ),
    )
    noise_scale: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Amount of noise to add during video generation (video mode only)",
        json_schema_extra=ui_field_config(
            order=7, component="noise", modes=["video"], is_load_param=True
        ),
    )
    noise_controller: bool = Field(
        default=True,
        description="Enable dynamic noise control during generation (video mode only)",
        json_schema_extra=ui_field_config(
            order=7, component="noise", modes=["video"], is_load_param=True
        ),
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method for the diffusion model.",
        json_schema_extra=ui_field_config(
            order=8, component="quantization", is_load_param=True
        ),
    )

    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(
            height=256,
            width=256,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
            default_temporal_interpolation_steps=0,
        ),
    }
