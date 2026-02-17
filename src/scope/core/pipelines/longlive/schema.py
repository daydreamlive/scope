from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    VACE_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..enums import Quantization
from ..utils import VaeType


class LongLiveConfig(BasePipelineConfig):
    pipeline_id = "longlive"
    pipeline_name = "LongLive"
    pipeline_description = (
        "A streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU. "
        "The model is trained using Self-Forcing on Wan2.1 1.3b with modifications to support smoother prompt "
        "switching and improved quality over longer time periods while maintaining fast generation."
    )
    docs_url = "https://github.com/daydreamlive/scope/blob/main/src/scope/core/pipelines/longlive/docs/usage.md"
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
            repo_id="Efficient-Large-Model/LongLive-1.3B",
            files=["models/longlive_base.pt", "models/lora.pt"],
        ),
    ]

    supports_cache_management = True
    supports_quantization = True
    min_dimension = 16
    modified = True

    # Configuration fields with UI metadata (order, component, modes)
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
    dummy_forcing: bool = Field(
        default=False,
        description=(
            "Enable Dummy Forcing optimization. Identifies attention heads that "
            "primarily attend to the current frame and routes them through a "
            "shorter attention path (sink + current only) for faster inference."
        ),
        json_schema_extra=ui_field_config(
            order=9, is_load_param=True, label="Dummy Forcing"
        ),
    )
    num_dummy: int = Field(
        default=180,
        ge=0,
        le=360,
        description=(
            "Number of attention heads to classify as dummy (out of 360 total = "
            "30 layers x 12 heads for Wan2.1 1.3B). Higher values = faster "
            "inference but potentially lower quality. Default: 180 (~50%)."
        ),
        json_schema_extra=ui_field_config(
            order=10, is_load_param=True, label="Dummy Heads"
        ),
    )
    df_local_context_length: int = Field(
        default=1,
        ge=1,
        le=5,
        description=(
            "Rolling window size (in AR blocks) for 'local' group heads. "
            "Higher = more context for local heads but less speedup. Default: 1."
        ),
        json_schema_extra=ui_field_config(
            order=11, is_load_param=True, label="DF Local Context"
        ),
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
