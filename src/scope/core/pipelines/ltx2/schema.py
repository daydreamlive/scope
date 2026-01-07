"""LTX2 pipeline configuration schema."""

from typing import ClassVar

from ..base_schema import BasePipelineConfig, ModeDefaults, height_field, width_field


class LTX2Config(BasePipelineConfig):
    """Configuration for LTX2 text-to-video pipeline.

    LTX2 is a high-quality video generation model that generates videos from text prompts.
    This is a non-autoregressive model that generates complete videos in one shot.
    """

    # Pipeline metadata
    pipeline_id: ClassVar[str] = "ltx2"
    pipeline_name: ClassVar[str] = "LTX2"
    pipeline_description: ClassVar[str] = (
        "High-quality text-to-video generation with LTX2 transformer"
    )
    pipeline_version: ClassVar[str] = "0.1.0"
    docs_url: ClassVar[str | None] = "https://github.com/Lightricks/LTX-2"
    estimated_vram_gb: ClassVar[float | None] = 32.0
    requires_models: ClassVar[bool] = True
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False

    # UI capability metadata
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = True
    min_dimension: ClassVar[int] = 64
    modified: ClassVar[bool] = False
    recommended_quantization_vram_threshold: ClassVar[float | None] = 32.0

    # Mode configuration - only supports text mode for now
    modes: ClassVar[dict[str, ModeDefaults]] = {"text": ModeDefaults(default=True)}

    # Prompt support
    supports_prompts: ClassVar[bool] = True

    # Resolution settings (LTX2 works best at these resolutions)
    # CRITICAL: Set to minimal values to fit in 96GB VRAM
    # Activations during denoising are NOT quantized and scale with resolution×frames
    # Even with FP8 weights, activations use 70+ GB at high settings
    height: int = height_field(default=512)
    width: int = width_field(default=768)

    # Number of frames to generate
    # Reduced to 33 frames (~1.3 seconds) to fit in 96GB VRAM
    # Memory formula: ~1.5GB per frame at 512x768 resolution
    num_frames: int = 33

    # Frame rate for video generation
    frame_rate: float = 24.0

    # Random seed for generation
    base_seed: int = 42

    # Memory optimization: Use FP8 quantization for transformer
    # According to official LTX-2 docs, this significantly reduces VRAM usage
    # Requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    use_fp8: bool = True
