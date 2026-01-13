"""
Common artifact definitions shared across built-in pipelines.

This module provides reusable artifact constants that multiple pipelines depend on.
Individual pipelines declare their artifacts via the `artifacts` ClassVar on their
config class, importing these constants as needed.
"""

from .artifacts import HuggingfaceRepoArtifact

# Common artifacts shared across built-in pipelines
WAN_1_3B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    files=["config.json", "Wan2.1_VAE.pth", "google"],
)

UMT5_ENCODER_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["umt5-xxl-enc-fp8_e4m3fn.safetensors"],
)

VACE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    files=["Wan2_1-VACE_module_1_3B_bf16.safetensors"],
)

VACE_14B_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Kijai/WanVideo_comfy",
    # Use BF16 version for CPU offloading compatibility (FP8 doesn't work on CPU)
    files=["Wan2_1-VACE_module_14B_bf16.safetensors"],
)

# Extra VAE artifacts (lightweight/alternative encoders)
LIGHTVAE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="lightx2v/Autoencoders",
    files=["lightvaew2_1.pth"],
)

TAE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="lightx2v/Autoencoders",
    files=["taew2_1.pth"],
)

LIGHTTAE_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="lightx2v/Autoencoders",
    files=["lighttaew2_1.pth"],
)
