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

    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16
    modified = True

    denoising_steps: list[int] = [750, 250]
    noise_scale: float = 0.7
    noise_controller: bool = True
    input_size: int = 4
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
    )

    modes = {
        "text": ModeDefaults(
            height=512,
            width=512,
            denoising_steps=[1000, 750],
        ),
        "video": ModeDefaults(default=True),
    }
