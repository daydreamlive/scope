from typing import ClassVar, Literal

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..enums import Quantization, VaeType


class TurboDiffusionConfig(BasePipelineConfig):
    pipeline_id: ClassVar[str] = "turbodiffusion"
    pipeline_name: ClassVar[str] = "TurboDiffusion"
    pipeline_description: ClassVar[str] = (
        "TurboDiffusion accelerates Wan2.1 1.3B video generation by 100-200x via "
        "rCM (Rectified Consistency Model) timestep distillation and SLA sparse attention. "
        "Generates complete videos in 1-4 denoising steps (non-streaming)."
    )
    estimated_vram_gb: ClassVar[float] = 8.0
    supports_quantization: ClassVar[bool] = True
    min_dimension: ClassVar[int] = 16
    supports_prompts: ClassVar[bool] = True

    artifacts: ClassVar[list] = [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="TurboDiffusion/TurboWan2.1-T2V-1.3B-480P",
            files=["TurboWan2.1-T2V-1.3B-480P.pth"],
        ),
    ]

    # Text-only mode (batch generation, not streaming)
    modes: ClassVar[dict] = {
        "text": ModeDefaults(default=True),
    }

    # --- UI fields ---

    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
        json_schema_extra=ui_field_config(order=1, is_load_param=True, label="VAE"),
    )
    height: int = Field(
        default=480,
        ge=1,
        description="Output height in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    width: int = Field(
        default=832,
        ge=1,
        description="Output width in pixels",
        json_schema_extra=ui_field_config(
            order=2, component="resolution", is_load_param=True
        ),
    )
    num_frames: int = Field(
        default=81,
        ge=1,
        le=201,
        description="Number of video frames to generate",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="Frames"),
    )
    num_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of rCM denoising steps (1-4 recommended). Fewer steps = faster but lower quality.",
        json_schema_extra=ui_field_config(order=4, label="Steps"),
    )
    sigma_max: float = Field(
        default=80.0,
        ge=1.0,
        le=200.0,
        description="Initial sigma for rCM sampling. Higher values produce more diverse outputs.",
        json_schema_extra=ui_field_config(order=5, label="Sigma Max"),
    )
    attention_type: Literal["original", "sla", "sagesla"] = Field(
        default="sagesla",
        description="Attention backend. 'sagesla' is fastest (requires SLA library), 'original' is the fallback.",
        json_schema_extra=ui_field_config(
            order=6, is_load_param=True, label="Attention"
        ),
    )
    sla_topk: float = Field(
        default=0.12,
        ge=0.05,
        le=0.5,
        description="Top-k ratio for SLA attention. Higher = better quality but slower. 0.15 recommended for quality.",
        json_schema_extra=ui_field_config(
            order=7, is_load_param=True, label="SLA Top-K"
        ),
    )
    base_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible generation",
        json_schema_extra=ui_field_config(order=8, is_load_param=True, label="Seed"),
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method for the diffusion model.",
        json_schema_extra=ui_field_config(
            order=9, component="quantization", is_load_param=True
        ),
    )
