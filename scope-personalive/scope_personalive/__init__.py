"""PersonaLive plugin for Daydream Scope.

This plugin provides the PersonaLive portrait animation pipeline for real-time
portrait animation from reference images and driving video.

Based on: https://github.com/GVCLab/PersonaLive
"""

import scope.core
from .pipeline import PersonaLivePipeline


@scope.core.hookimpl
def register_pipelines(register):
    """Register the PersonaLive pipeline."""
    register(PersonaLivePipeline)


@scope.core.hookimpl
def register_artifacts(register):
    """Register PersonaLive model artifacts for download."""
    from scope.server.artifacts import HuggingfaceRepoArtifact

    register(
        "personalive",
        [
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
    )


@scope.core.hookimpl
def register_routes(app):
    """Register PersonaLive-specific API routes."""
    from .routes import register_personalive_routes

    register_personalive_routes(app)


__all__ = ["PersonaLivePipeline"]
