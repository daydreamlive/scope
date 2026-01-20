"""Pydantic schemas for FastAPI application."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Import enums from torch-free module to avoid loading torch at CLI startup
from scope.core.pipelines.enums import Quantization, VaeType

# Default values for pipeline load params (duplicated from pipeline configs to avoid
# importing torch-dependent modules). These should match the defaults in:
# - StreamDiffusionV2Config: height=512, width=512, base_seed=42
# - LongLiveConfig: height=320, width=576, base_seed=42
# - KreaRealtimeVideoConfig: height=320, width=576, base_seed=42
_STREAMDIFFUSIONV2_HEIGHT = 512
_STREAMDIFFUSIONV2_WIDTH = 512
_LONGLIVE_HEIGHT = 320
_LONGLIVE_WIDTH = 576
_KREA_HEIGHT = 320
_KREA_WIDTH = 576
_DEFAULT_SEED = 42
_DEFAULT_VAE_TYPE = VaeType.WAN


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
    lora_scales: list["LoRAScaleUpdate"] | None = Field(
        default=None,
        description="Update scales for loaded LoRA adapters. Each entry updates a specific adapter by path.",
    )
    spout_sender: "SpoutConfig | None" = Field(
        default=None,
        description="Spout output configuration for sending frames to external apps",
    )
    spout_receiver: "SpoutConfig | None" = Field(
        default=None,
        description="Spout input configuration for receiving frames from external apps",
    )
    vace_enabled: bool | None = Field(
        default=None,
        description="Enable VACE (Video All-In-One Creation and Editing) for reference image conditioning and structural guidance. Must be enabled at pipeline load time for VACE to be available.",
    )
    vace_ref_images: list[str] | None = Field(
        default=None,
        description="List of reference image file paths for VACE conditioning. Images should be located in the assets directory (at the same level as the models directory).",
    )
    vace_use_input_video: bool | None = Field(
        default=None,
        description="When enabled in Video input mode, the input video is used for VACE conditioning. When disabled, the input video is used for latent initialization instead, allowing reference images to be used while in Video input mode.",
    )
    vace_context_scale: float = Field(
        default=1.0,
        description="Scaling factor for VACE hint injection. Higher values make reference images more influential.",
        ge=0.0,
        le=2.0,
    )
    pipeline_ids: list[str] | None = Field(
        default=None,
        description="List of pipeline IDs to execute in a chain. If not provided, uses the currently loaded pipeline.",
    )
    first_frame_image: str | None = Field(
        default=None,
        description="Path to first frame reference image for extension mode. When provided alone, enables 'firstframe' mode (reference at start, generate continuation). When provided with last_frame_image, enables 'firstlastframe' mode (references at both ends). Images should be located in the assets directory.",
    )
    last_frame_image: str | None = Field(
        default=None,
        description="Path to last frame reference image for extension mode. When provided alone, enables 'lastframe' mode (generate lead-up, reference at end). When provided with first_frame_image, enables 'firstlastframe' mode (references at both ends). Images should be located in the assets directory.",
    )
    images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for non-VACE visual conditioning",
    )


class SpoutConfig(BaseModel):
    """Configuration for Spout sender/receiver."""

    enabled: bool = Field(default=False, description="Enable Spout")
    name: str = Field(default="", description="Spout sender name")


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
    sessionId: str = Field(..., description="Unique session ID for this connection")


class IceServerConfig(BaseModel):
    """ICE server configuration for WebRTC."""

    urls: str | list[str] = Field(..., description="STUN/TURN server URL(s)")
    username: str | None = Field(default=None, description="Username for TURN server")
    credential: str | None = Field(
        default=None, description="Credential for TURN server"
    )


class IceServersResponse(BaseModel):
    """Response containing ICE server configuration."""

    iceServers: list[IceServerConfig] = Field(
        ..., description="List of ICE servers for WebRTC connection"
    )


class IceCandidateInit(BaseModel):
    """Individual ICE candidate initialization data."""

    candidate: str = Field(..., description="ICE candidate string")
    sdpMid: str | None = Field(default=None, description="Media stream ID")
    sdpMLineIndex: int | None = Field(
        default=None, description="Media line index in SDP"
    )


class IceCandidateRequest(BaseModel):
    """Request to add ICE candidate(s) to an existing session."""

    candidates: list[IceCandidateInit] = Field(
        ..., description="List of ICE candidates to add"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: str = Field(None, description="Additional error details")


class HardwareInfoResponse(BaseModel):
    """Hardware information response schema."""

    vram_gb: float | None = Field(
        default=None, description="Total VRAM in GB (None if CUDA not available)"
    )
    spout_available: bool = Field(
        default=False,
        description="Whether Spout is available (Windows only, not WSL)",
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

    Defaults match StreamDiffusionV2Config values.
    """

    height: int = Field(
        default=_STREAMDIFFUSIONV2_HEIGHT,
        description="Target video height",
        ge=64,
        le=2048,
    )
    width: int = Field(
        default=_STREAMDIFFUSIONV2_WIDTH,
        description="Target video width",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=_DEFAULT_SEED,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )
    vace_enabled: bool = Field(
        default=True,
        description="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, input video in Video input mode can be used for VACE conditioning. When disabled, video uses faster regular encoding for latent initialization.",
    )
    vae_type: VaeType = Field(
        default=_DEFAULT_VAE_TYPE,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality), 'tae' is a tiny autoencoder for fast preview quality, 'lighttae' is LightTAE with WanVAE normalization.",
    )


class PassthroughLoadParams(PipelineLoadParams):
    """Load parameters for Passthrough pipeline."""

    pass


class LongLiveLoadParams(LoRAEnabledLoadParams):
    """Load parameters for LongLive pipeline.

    Defaults match LongLiveConfig values.
    """

    height: int = Field(
        default=_LONGLIVE_HEIGHT,
        description="Target video height",
        ge=16,
        le=2048,
    )
    width: int = Field(
        default=_LONGLIVE_WIDTH,
        description="Target video width",
        ge=16,
        le=2048,
    )
    seed: int = Field(
        default=_DEFAULT_SEED,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=None,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )
    vace_enabled: bool = Field(
        default=True,
        description="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, input video in Video input mode can be used for VACE conditioning. When disabled, video uses faster regular encoding for latent initialization.",
    )
    vae_type: VaeType = Field(
        default=_DEFAULT_VAE_TYPE,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality), 'tae' is a tiny autoencoder for fast preview quality, 'lighttae' is LightTAE with WanVAE normalization.",
    )


class KreaRealtimeVideoLoadParams(LoRAEnabledLoadParams):
    """Load parameters for KreaRealtimeVideo pipeline.

    Defaults match KreaRealtimeVideoConfig values.
    """

    height: int = Field(
        default=_KREA_HEIGHT,
        description="Target video height",
        ge=64,
        le=2048,
    )
    width: int = Field(
        default=_KREA_WIDTH,
        description="Target video width",
        ge=64,
        le=2048,
    )
    seed: int = Field(
        default=_DEFAULT_SEED,
        description="Random seed for generation",
        ge=0,
    )
    quantization: Quantization | None = Field(
        default=Quantization.FP8_E4M3FN,
        description="Quantization method to use for diffusion model. If None, no quantization is applied.",
    )
    vace_enabled: bool = Field(
        default=True,
        description="Enable VACE (Video All-In-One Creation and Editing) support for reference image conditioning and structural guidance. When enabled, input video in Video input mode can be used for VACE conditioning. When disabled, video uses faster regular encoding for latent initialization.",
    )
    vae_type: VaeType = Field(
        default=_DEFAULT_VAE_TYPE,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality), 'tae' is a tiny autoencoder for fast preview quality, 'lighttae' is LightTAE with WanVAE normalization.",
    )


class PipelineLoadRequest(BaseModel):
    """Pipeline load request schema."""

    pipeline_ids: list[str] = Field(..., description="List of pipeline IDs to load")
    load_params: (
        StreamDiffusionV2LoadParams
        | PassthroughLoadParams
        | LongLiveLoadParams
        | KreaRealtimeVideoLoadParams
        | None
    ) = Field(
        default=None,
        description="Pipeline-specific load parameters (applies to all pipelines)",
    )


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


class AssetFileInfo(BaseModel):
    """Metadata for an available asset file on disk."""

    name: str
    path: str
    size_mb: float
    folder: str | None = None
    type: str  # "image" or "video"
    created_at: float  # Unix timestamp


class AssetsResponse(BaseModel):
    """Response containing all discoverable asset files."""

    assets: list[AssetFileInfo]
