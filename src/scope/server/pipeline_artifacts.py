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

# Pipeline-specific artifacts
PIPELINE_ARTIFACTS = {
    "streamdiffusionv2": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="jerryfeng/StreamDiffusionV2",
            files=["wan_causal_dmd_v2v/model.pt"],
        ),
    ],
    "longlive": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="Efficient-Large-Model/LongLive-1.3B",
            files=["models/longlive_base.pt", "models/lora.pt"],
        ),
    ],
    "krea-realtime-video": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
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
        HuggingfaceRepoArtifact(
            repo_id="JaydenLu666/Reward-Forcing-T2V-1.3B",
            files=["rewardforcing.pt"],
        ),
    ],
}
