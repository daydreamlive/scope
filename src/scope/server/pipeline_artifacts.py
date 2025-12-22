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

# Pipeline-specific artifacts
PIPELINE_ARTIFACTS = {
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
    "personalive": [
        # Base model with CLIP image encoder (SD fine-tuned to accept CLIP image embeddings)
        HuggingfaceRepoArtifact(
            repo_id="lambdalabs/sd-image-variations-diffusers",
            files=["image_encoder", "unet", "model_index.json"],
            local_dir="PersonaLive/pretrained_weights/sd-image-variations-diffusers",
        ),
        # Improved VAE fine-tuned on MSE loss
        HuggingfaceRepoArtifact(
            repo_id="stabilityai/sd-vae-ft-mse",
            files=["config.json", "diffusion_pytorch_model.safetensors"],
            local_dir="PersonaLive/pretrained_weights/sd-vae-ft-mse",
        ),
        # PersonaLive-specific weights: denoising_unet, reference_unet, motion modules, etc.
        # Files are at pretrained_weights/personalive/ in the repo
        # See: https://huggingface.co/huaichang/PersonaLive/tree/main/pretrained_weights/personalive
        HuggingfaceRepoArtifact(
            repo_id="huaichang/PersonaLive",
            files=[
                "pretrained_weights/personalive/denoising_unet.pth",
                "pretrained_weights/personalive/reference_unet.pth",
                "pretrained_weights/personalive/motion_encoder.pth",
                "pretrained_weights/personalive/motion_extractor.pth",
                "pretrained_weights/personalive/pose_guider.pth",
                "pretrained_weights/personalive/temporal_module.pth",
            ],
            local_dir="PersonaLive",
        ),
    ],
}
