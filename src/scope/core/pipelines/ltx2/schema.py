"""LTX2 pipeline configuration schema."""

from typing import ClassVar

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, height_field, width_field
from ..enums import Quantization


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
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="Lightricks/LTX-2",
            files=[
                "ltx-2-19b-distilled.safetensors",
                "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            ],
        ),
        HuggingfaceRepoArtifact(
            repo_id="google/gemma-3-12b-it",
            files=[
                "config.json",
                "generation_config.json",
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
                "model.safetensors.index.json",
                "processor_config.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ],
        ),
    ]
    supports_lora: ClassVar[bool] = False
    supports_vace: ClassVar[bool] = False

    # UI capability metadata
    supports_cache_management: ClassVar[bool] = False
    supports_kv_cache_bias: ClassVar[bool] = False
    supports_quantization: ClassVar[bool] = True
    min_dimension: ClassVar[int] = 64
    modified: ClassVar[bool] = False
    recommended_quantization_vram_threshold: ClassVar[float | None] = 32.0
    # LTX2 is bidirectional (not autoregressive), so randomize seed is useful
    supports_randomize_seed: ClassVar[bool] = True
    # LTX2 supports configurable number of frames per generation
    supports_num_frames: ClassVar[bool] = True

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

    # Memory optimization: Quantization for transformer weights
    # - fp8_e4m3fn: ~2x memory reduction (requires SM >= 8.9 Ada)
    # - nvfp4: ~4x memory reduction (requires SM >= 10.0 Blackwell)
    # - None: Full precision BF16
    # Requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    quantization: Quantization | None = Field(
        default=Quantization.FP8_E4M3FN,
        description=(
            "Quantization method for transformer weights. "
            "fp8_e4m3fn reduces memory ~2x (requires Ada GPU SM >= 8.9). "
            "nvfp4 reduces memory ~4x (requires Blackwell GPU SM >= 10.0). "
            "None uses full precision BF16."
        ),
    )

    # Deprecated: Use 'quantization' field instead
    # Kept for backwards compatibility
    use_fp8: bool | None = Field(
        default=None,
        description="Deprecated: Use 'quantization' field instead.",
    )

    # FFN chunking for memory-efficient inference
    # FFN layers expand hidden dimensions by 4x, creating massive intermediate tensors.
    # By processing the sequence in chunks, we reduce peak activation memory by ~10x.
    # Set to None to disable chunking.
    ffn_chunk_size: int | None = Field(
        default=4096,
        description=(
            "Chunk size for FFN processing. Smaller values use less memory but "
            "have more kernel launch overhead. Set to None to disable chunking. "
            "Default 4096 reduces activation memory from ~50GB to ~5GB."
        ),
    )

    # Randomize seed on every generation
    # LTX2 is bidirectional (not autoregressive), so each chunk is independent.
    # With a fixed seed, the same chunk is regenerated unless the prompt changes.
    # Enable this to get varied outputs between chunks.
    randomize_seed: bool = False
