from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults
from ..utils import VaeType


class MemFlowConfig(BasePipelineConfig):
    pipeline_id = "memflow"
    pipeline_name = "MemFlow"
    pipeline_description = (
        "A streaming pipeline and autoregressive video diffusion model from Kling. "
        "The model is trained using Self-Forcing on Wan2.1 1.3b based on the LongLive training and "
        "inference pipeline with the additions of a memory bank to improve long context consistency."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/memflow/docs/usage.md"
    estimated_vram_gb = 20.0
    requires_models = True
    supports_lora = True
    supports_vace = True

    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16
    modified = True

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
            height=512,
            width=512,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
        ),
    }
