"""
Defines which artifacts each pipeline requires.

Built-in pipelines define their artifacts here. Plugin pipelines can register
their artifacts via the `register_artifacts` hook.
"""

from .artifacts import Artifact, HuggingfaceRepoArtifact

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

# Built-in pipeline artifacts
_BUILTIN_PIPELINE_ARTIFACTS: dict[str, list[Artifact]] = {
    "streamdiffusionv2": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="jerryfeng/StreamDiffusionV2",
            files=["wan_causal_dmd_v2v/model.pt"],
        ),
    ],
    "longlive": [
        WAN_1_3B_ARTIFACT,
        UMT5_ENCODER_ARTIFACT,
        VACE_ARTIFACT,
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
        VACE_ARTIFACT,
        HuggingfaceRepoArtifact(
            repo_id="JaydenLu666/Reward-Forcing-T2V-1.3B",
            files=["rewardforcing.pt"],
        ),
    ],
}


def get_pipeline_artifacts() -> dict[str, list[Artifact]]:
    """Get all pipeline artifacts including those registered by plugins.

    Returns:
        Dictionary mapping pipeline_id to list of artifacts
    """
    from scope.core.plugins import load_plugins, register_plugin_artifacts

    # Ensure plugins are loaded (idempotent if already loaded)
    load_plugins()

    # Start with built-in artifacts
    all_artifacts = dict(_BUILTIN_PIPELINE_ARTIFACTS)

    # Register plugin artifacts
    register_plugin_artifacts(all_artifacts)

    return all_artifacts


# Legacy alias for backward compatibility
PIPELINE_ARTIFACTS = _BUILTIN_PIPELINE_ARTIFACTS
