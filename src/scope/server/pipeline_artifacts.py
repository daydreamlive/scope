"""
Defines which artifacts each pipeline requires.
"""

from .artifacts import HuggingfaceRepoArtifact

# Common artifacts shared across pipelines
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

# LTX2 artifacts
LTX2_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="Lightricks/LTX-2",
    files=[
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-spatial-upsampler-x2-1.0.safetensors",
    ],
)

GEMMA_TEXT_ENCODER_ARTIFACT = HuggingfaceRepoArtifact(
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
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    ],
)

# Pipeline-specific artifacts
PIPELINE_ARTIFACTS = {
    "streamdiffusionv2": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="jerryfeng/StreamDiffusionV2",
            files=["wan_causal_dmd_v2v/model.pt"],
        ),
    ],
    "longlive": [
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
    ],
    "krea-realtime-video": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="Wan-AI/Wan2.1-T2V-14B",
            files=["config.json"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="krea/krea-realtime-video",
            files=["krea-realtime-video-14b.safetensors"],
        ),
    ],
    "reward-forcing": [
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
    ],
    "memflow": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="KlingTeam/MemFlow",
            files=["base.pt", "lora.pt"],
        ),
    ],
    "ltx2": [
        LTX2_ARTIFACT,
        GEMMA_TEXT_ENCODER_ARTIFACT,
    ],
}
