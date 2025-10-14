"""Pydantic schemas for FastAPI application."""

from enum import Enum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy")
    timestamp: str


class Parameters(BaseModel):
    """Parameters for WebRTC session."""

    prompts: list[str] | None = Field(default=None, description="Prompt list")
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


class PipelineStatusEnum(str, Enum):
    """Pipeline status enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


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
