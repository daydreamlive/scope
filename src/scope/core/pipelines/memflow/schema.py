from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
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
            repo_id="KlingTeam/MemFlow",
            files=["base.pt", "lora.pt"],
        ),
    ]

    min_dimension = 16
    modified = True
    supports_quantization = True

    height: int = Field(
        default=320,
        json_schema_extra={
            "ui:category": "resolution",
            "ui:order": 1,
            "ui:label": "Height",
        },
    )
    width: int = Field(
        default=576,
        json_schema_extra={
            "ui:category": "resolution",
            "ui:order": 2,
            "ui:label": "Width",
        },
    )
    denoising_steps: list[int] = Field(
        default=[1000, 750, 500, 250],
        json_schema_extra={
            "ui:category": "generation",
            "ui:order": 1,
            "ui:widget": "denoisingSteps",
            "ui:label": "Denoising Steps",
        },
    )
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use for encoding/decoding. 'wan' is the full VAE with best quality. 'lightvae' is 75% pruned for faster performance but lower quality. 'tae' is a tiny autoencoder for fast preview quality. 'lighttae' is LightTAE with WanVAE normalization for faster performance with consistent latent space.",
        json_schema_extra={
            "ui:category": "generation",
            "ui:order": 2,
            "ui:label": "VAE Type",
        },
    )

    modes = {
        "text": ModeDefaults(
            default=True,
        ),
        "video": ModeDefaults(
            height=512,
            width=512,
            noise_scale=0.7,
            noise_controller=True,
            denoising_steps=[1000, 750],
        ),
    }
