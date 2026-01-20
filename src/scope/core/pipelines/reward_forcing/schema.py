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
