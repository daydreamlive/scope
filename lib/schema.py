"""Pydantic schemas for FastAPI application."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class Quantization(str, Enum):
    """Quantization method enumeration."""

    FP8_E4M3FN = "fp8_e4m3fn"


class PipelineLoadParams(BaseModel):
    """Base class for pipeline load parameters."""

    pass


class StreamDiffusionV2LoadParams(PipelineLoadParams):
    """Load parameters for StreamDiffusion V2 pipeline."""

    height: int = Field(default=512, description="Target video height", ge=64, le=2048)
    width: int = Field(default=512, description="Target video width", ge=64, le=2048)
    seed: int = Field(default=42, description="Random seed for generation", ge=0)


class PassthroughLoadParams(PipelineLoadParams):
    """Load parameters for Passthrough pipeline."""

    pass


class VodLoadParams(PipelineLoadParams):
    """Load parameters for VOD pipeline."""

    pass


class LongLiveLoadParams(PipelineLoadParams):
    """Load parameters for LongLive pipeline."""

    height: int = Field(default=320, description="Target video height", ge=16, le=2048)
    width: int = Field(default=576, description="Target video width", ge=16, le=2048)
    seed: int = Field(default=42, description="Random seed for generation", ge=0)


class KreaRealtimeVideoLoadParams(PipelineLoadParams):
    """Load parameters for KreaRealtimeVideo pipeline."""

    height: int = Field(default=512, description="Target video height", ge=64, le=2048)
    width: int = Field(default=512, description="Target video width", ge=64, le=2048)
    seed: int = Field(default=42, description="Random seed for generation", ge=0)
    quantization: Quantization | None = Field(
        default=Quantization.FP8_E4M3FN,
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
        | VodLoadParams
        | LongLiveLoadParams
        | KreaRealtimeVideoLoadParams
        | None
    ) = Field(default=None, description="Pipeline-specific load parameters")


class PipelineStatusResponse(BaseModel):
    """Pipeline status response schema."""

    status: PipelineStatusEnum = Field(..., description="Current pipeline status")
    pipeline_id: str | None = Field(default=None, description="ID of loaded pipeline")
    load_params: dict | None = Field(
        default=None, description="Load parameters used when loading the pipeline"
    )
    error: str | None = Field(
        default=None, description="Error message if status is error"
    )
