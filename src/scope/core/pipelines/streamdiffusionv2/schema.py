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


class StreamDiffusionV2Config(BasePipelineConfig):
    pipeline_id = "streamdiffusionv2"
    pipeline_name = "StreamDiffusionV2"
    pipeline_description = (
        "A streaming pipeline and autoregressive video diffusion model from the creators of the original "
        "StreamDiffusion project. The model is trained using Self-Forcing on Wan2.1 1.3b with modifications "
        "to support streaming."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/streamdiffusionv2/docs/usage.md"
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
            repo_id="jerryfeng/StreamDiffusionV2",
            files=["wan_causal_dmd_v2v/model.pt"],
        ),
    ]

    min_dimension = 16
    modified = True
    supports_quantization = True

    denoising_steps: list[int] = Field(
        default=[750, 250],
        json_schema_extra={
            "ui:category": "generation",
            "ui:order": 1,
            "ui:widget": "denoisingSteps",
            "ui:label": "Denoising Steps",
        },
    )
    noise_scale: float = Field(
        default=0.7,
        json_schema_extra={
            "ui:category": "noise",
            "ui:order": 2,
            "ui:label": "Noise Scale",
            "ui:showIf": {"field": "input_mode", "eq": "video"},
        },
    )
    noise_controller: bool = Field(
        default=True,
        json_schema_extra={
            "ui:category": "noise",
            "ui:order": 1,
            "ui:label": "Noise Controller",
            "ui:showIf": {"field": "input_mode", "eq": "video"},
        },
    )
    input_size: int = Field(
        default=4,
        json_schema_extra={
            "ui:hidden": True,  # Internal field, not user-facing
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
            height=512,
            width=512,
            denoising_steps=[1000, 750],
        ),
        "video": ModeDefaults(
            default=True,
            noise_scale=0.7,
            noise_controller=True,
        ),
    }
