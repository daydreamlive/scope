"""Pydantic schemas for FastAPI application."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from scope.core.pipelines.schema import (
    KreaRealtimeVideoConfig,
    LongLiveConfig,
    RewardForcingConfig,
    StreamDiffusionV2Config,
)
from scope.core.pipelines.utils import Quantization


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy")
    timestamp: str


class PromptItem(BaseModel):
    """Individual prompt with weight for blending."""

    text: str = Field(..., description="Prompt text")
    weight: float = Field(
        default=1.0, ge=0.0, description="Weight for blending (must be non-negative)"
    )


class PromptTransition(BaseModel):
    """Configuration for transitioning between prompt blends over time.

    This controls temporal interpolation - how smoothly prompts transition
    across multiple generation frames, distinct from spatial blending of
    multiple prompts within a single frame.
    """

    target_prompts: list[PromptItem] = Field(
        ..., description="Target prompt blend to interpolate to"
    )
    num_steps: int = Field(
        default=4,
        ge=0,
        description="Number of generation calls to transition over (0 = instant, 4 is default)",
    )
    temporal_interpolation_method: Literal["linear", "slerp"] = Field(
        default="linear",
        description="Method for temporal interpolation between blends across frames",
    )


class Parameters(BaseModel):
    """Parameters for WebRTC session."""

    input_mode: Literal["text", "video"] | None = Field(
        default=None,
        description="Input mode for the stream: 'text' for text-to-video, 'video' for video-to-video",
    )
    prompts: list[PromptItem] | None = Field(
        default=None,
        description="List of prompts with weights for spatial blending within a single frame",
    )
    prompt_interpolation_method: Literal["linear", "slerp"] = Field(
        default="linear",
        description="Spatial interpolation method for blending multiple prompts: linear (weighted average) or slerp (spherical)",
    )
    transition: PromptTransition | None = Field(
        default=None,
        description="Optional transition to smoothly interpolate from current prompts to target prompts over multiple frames. "
        "When provided, the transition.target_prompts will become the new prompts after the transition completes, "
        "and this field takes precedence over the 'prompts' field for initiating the transition.",
    )
    noise_scale: float | None = Field(
        default=None, description="Noise scale (0.0-1.0)", ge=0.0, le=1.0
    )
    noise_controller: bool | None = Field(
        default=None,
        description="Enable automatic noise scale adjustment based on motion detection",
    )
    denoising_step_list: list[int] | None = Field(
        default=None, description="Denoising step list"
    )
    manage_cache: bool | None = Field(
        default=None,
        description="Enable automatic cache management for parameter updates",
    )
    reset_cache: bool | None = Field(default=None, description="Trigger a cache reset")
    kv_cache_attention_bias: float | None = Field(
        default=None,
        description="Controls how much to rely on past frames in the cache during generation. A lower value can help mitigate error accumulation and prevent repetitive motion. Uses log scale: 1.0 = full reliance on past frames, smaller values = less reliance on past frames. Typical values: 0.3-0.7 for moderate effect, 0.1-0.2 for strong effect.",
        ge=0.01,
        le=1.0,
    )
    compression_alpha: float | None = Field(
        default=None,
        description="EMA coefficient for sink token compression (Reward-Forcing pipeline). "
        "Controls how much weight is given to historical context vs recent frames. "
        "Higher values (0.999) = more stable, but prone to semantic drift in long videos. "
        "Lower values (0.9-0.95) = adapts faster, reduces drift but may lose long-term consistency. "
        "Typical values: 0.999 (default), 0.99 (balanced), 0.95 (aggressive anti-drift).",
        ge=0.0,
        le=1.0,
    )
    semantic_refresh_interval: int | None = Field(
        default=None,
        description="Interval (in frames) at which to refresh the cross-attention cache to combat semantic drift. "
        "When set, the text conditioning is periodically re-applied to prevent drift in long videos. "
        "Set to 0 to disable (default). Recommended values: 100-500 for long videos.",
        ge=0,
    )
    lora_scales: list["LoRAScaleUpdate"] | None = Field(
        default=None,
        description="Update scales for loaded LoRA adapters. Each entry updates a specific adapter by path.",
    )


class WebRTCOfferRequest(BaseModel):
    """WebRTC offer request schema."""

    sdp: str = Field(..., description="Session Description Protocol offer")
    type: str = Field(..., description="SDP type (should be 'offer')")
    initialParameters: Parameters | None = Field(
        default=None, description="Initial parameters for the session"
    )


class WebRTCOfferResponse(BaseModel):
    """WebRTC offer response schema."""

    sdp: str = Field(..., description="Session Description Protocol answer")
    type: str = Field(..., description="SDP type (should be 'answer')")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Additional error details")


class HardwareInfoResponse(BaseModel):
    """Hardware information response schema."""

    vram_gb: float | None = Field(
        default=None, description="Total VRAM in GB (None if CUDA not available)"
    )


class PipelineStatusEnum(str, Enum):
    """Pipeline status enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class LoRAMergeMode(str, Enum):
    """LoRA merge mode enumeration."""

    RUNTIME_PEFT = "runtime_peft"
    PERMANENT_MERGE = "permanent_merge"


class LoRAConfig(BaseModel):
    """Configuration for a LoRA (Low-Rank Adaptation) adapter."""

    path: str = Field(
        ...,
        description=(
            "Local path to LoRA weights file (.safetensors, .bin, .pt). "
            "Typically under models/lora/."
        ),
    )
    scale: float = Field(
        default=1.0,
        ge=-10.0,
        le=10.0,
        description=(
            "Adapter strength/weight (-10.0 to 10.0, 0.0 = disabled, 1.0 = full strength)."
        ),
    )
    merge_mode: LoRAMergeMode | None = Field(
        default=None,
        description=(
            "Optional merge strategy for this specific LoRA. "
            "If not specified, uses the pipeline's default lora_merge_mode. "
            "Permanent merge offers maximum FPS but no runtime updates; "
            "runtime_peft offers instant updates at reduced FPS."
        ),
    )


class LoRAScaleUpdate(BaseModel):
    """Update scale for a loaded LoRA adapter."""

    path: str = Field(
        ..., description="Path of the LoRA to update (must match loaded path)"
    )
    scale: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="New adapter strength/weight (-10.0 to 10.0, 0.0 = disabled, 1.0 = full strength).",
    )


class PipelineLoadParams(BaseModel):
    """Base class for pipeline load parameters."""

    pass


class LoRAEnabledLoadParams(PipelineLoadParams):
    """Base class for load params that support LoRA."""

    loras: list[LoRAConfig] | None = Field(
        default=None, description="Optional list of LoRA adapter configurations."
    )
    lora_merge_mode: LoRAMergeMode = Field(
        default=LoRAMergeMode.PERMANENT_MERGE,
        description=(
            "LoRA merge strategy. Permanent merge offers maximum FPS but no runtime updates; "
            "runtime_peft offers instant updates at reduced FPS."
        ),
    )


class StreamDiffusionV2LoadParams(LoRAEnabledLoadParams):
    """Load parameters for StreamDiffusion V2 pipeline.

    Defaults are derived from StreamDiffusionV2Config to ensure consistency.
    """

    height: int = Field(
        default=StreamDiffusionV2Config.model_fields["height"].default,
        description="Target video height",
        ge=64,
        le=2048,
    )
    width: int = Field(
        default=StreamDiffusionV2Config.model_fields["width"].default,
        description="Target video width",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=StreamDiffusionV2Config.model_fields["base_seed"].default,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )


class PassthroughLoadParams(PipelineLoadParams):
    """Load parameters for Passthrough pipeline."""

    pass


class LongLiveLoadParams(LoRAEnabledLoadParams):
    """Load parameters for LongLive pipeline.

    Defaults are derived from LongLiveConfig to ensure consistency.
    """

    height: int = Field(
        default=LongLiveConfig.model_fields["height"].default,
        description="Target video height",
        ge=16,
        le=2048,
    )
    width: int = Field(
        default=LongLiveConfig.model_fields["width"].default,
        description="Target video width",
        ge=16,
        le=2048,
    )
    seed: int = Field(
        default=LongLiveConfig.model_fields["base_seed"].default,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )


class KreaRealtimeVideoLoadParams(LoRAEnabledLoadParams):
    """Load parameters for KreaRealtimeVideo pipeline.

    Defaults are derived from KreaRealtimeVideoConfig to ensure consistency.
    """

    height: int = Field(
        default=KreaRealtimeVideoConfig.model_fields["height"].default,
        description="Target video height",
        ge=64,
        le=2048,
    )
    width: int = Field(
        default=KreaRealtimeVideoConfig.model_fields["width"].default,
        description="Target video width",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=KreaRealtimeVideoConfig.model_fields["base_seed"].default,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=Quantization.FP8_E4M3FN,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )


class RewardForcingLoadParams(LoRAEnabledLoadParams):
    """Load parameters for Reward-Forcing pipeline.

    Reward-Forcing is a training method that enables few-step video generation
    by learning from reward signals. The distilled model can generate high-quality
    videos in just 4 denoising steps (vs. 50+ for standard diffusion).

    Defaults are derived from RewardForcingConfig to ensure consistency.

    Reference: https://github.com/JaydenLu666/Reward-Forcing
    """

    height: int = Field(
        default=RewardForcingConfig.model_fields["height"].default,
        description="Target video height",
        ge=16,
        le=2048,
    )
    width: int = Field(
        default=RewardForcingConfig.model_fields["width"].default,
        description="Target video width",
        ge=16,
        le=2048,
    )
    seed: int = Field(
        default=RewardForcingConfig.model_fields["base_seed"].default,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )


class PipelineLoadRequest(BaseModel):
    """Pipeline load request schema."""

    pipeline_id: str = Field(
        default="streamdiffusionv2", description="ID of pipeline to load"
    )
    load_params: (
        StreamDiffusionV2LoadParams
        | PassthroughLoadParams
        | LongLiveLoadParams
        | KreaRealtimeVideoLoadParams
        | RewardForcingLoadParams
        | None
    ) = Field(default=None, description="Pipeline-specific load parameters")


class PipelineStatusResponse(BaseModel):
    """Pipeline status response schema."""

    status: PipelineStatusEnum = Field(..., description="Current pipeline status")
    pipeline_id: str | None = Field(default=None, description="ID of loaded pipeline")
    load_params: dict | None = Field(
        default=None, description="Load parameters used when loading the pipeline"
    )
    loaded_lora_adapters: list[dict] | None = Field(
        default=None,
        description=(
            "Information about currently loaded LoRA adapters (path and scale). "
            "Used by the frontend to decide which adapters can be updated at runtime."
        ),
    )
    error: str | None = Field(
        default=None, description="Error message if status is error"
    )


class PipelineSchemasResponse(BaseModel):
    """Response containing schemas for all available pipelines.

    Each pipeline entry contains the output of get_schema_with_metadata()
    plus additional mode information.
    """

    pipelines: dict = Field(..., description="Pipeline schemas keyed by pipeline ID")
