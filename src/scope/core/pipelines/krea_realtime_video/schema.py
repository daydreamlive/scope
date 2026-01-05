from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults
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
    requires_models = True
    supports_lora = True

    supports_cache_management = True
    supports_kv_cache_bias = True
    supports_quantization = True
    min_dimension = 16
    modified = True
    recommended_quantization_vram_threshold = 40.0

    default_temporal_interpolation_method = "linear"
    default_temporal_interpolation_steps = 4

    height: int = 320
    width: int = 576
    denoising_steps: list[int] = [1000, 750, 500, 250]
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
    )

    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(
            height=256,
            width=256,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
        ),
    }
