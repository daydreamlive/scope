"""CLI script for converting PersonaLive models to TensorRT.

This script handles the full conversion pipeline:
1. Load PyTorch models
2. Create bundled UNetWork model
3. Export to ONNX
4. Optimize ONNX
5. Build TensorRT engine

Usage:
    convert-personalive-trt --model-dir ./models --height 512 --width 512
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def log_gpu_memory(msg: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory {msg}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PersonaLive models to TensorRT engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the models directory containing PersonaLive weights",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output video height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output video width",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 instead of FP16 precision",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (use existing ONNX files)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use",
    )
    return parser.parse_args()


def load_models(model_dir: Path, device: torch.device, dtype: torch.dtype):
    """Load all PersonaLive models with aggressive memory management.

    Args:
        model_dir: Base model directory.
        device: Target device.
        dtype: Data type for models (used for TensorRT, ONNX export uses FP32).

    Returns:
        Dictionary containing all loaded models and config.
    """
    from ..modules import (
        DDIMScheduler,
        MotEncoder,
        PoseGuider,
    )
    # Use explicit reference UNet for TensorRT export
    from .unet_explicit import UNet3DConditionModelExplicit

    def cleanup():
        """Aggressive memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info("Loading PersonaLive models...")
    cleanup()

    # Paths
    personalive_dir = model_dir / "PersonaLive" / "pretrained_weights"
    pretrained_base_path = personalive_dir / "sd-image-variations-diffusers"
    vae_path = personalive_dir / "sd-vae-ft-mse"
    weights_path = personalive_dir / "personalive"

    # Load model config for UNet kwargs
    config_path = Path(__file__).parent.parent / "model.yaml"
    model_config = OmegaConf.load(config_path)
    unet_kwargs = OmegaConf.to_container(model_config.unet_additional_kwargs)
    sched_kwargs = OmegaConf.to_container(model_config.noise_scheduler_kwargs)

    # Use FP16 like official implementation - autocast handles dtype during export
    export_dtype = dtype

    # Load pose guider
    logger.info("Loading PoseGuider...")
    pose_guider = PoseGuider().to(device=device, dtype=export_dtype)
    pose_guider_path = weights_path / "pose_guider.pth"
    if pose_guider_path.exists():
        state_dict = torch.load(pose_guider_path, map_location="cpu")
        pose_guider.load_state_dict(state_dict)
        del state_dict
    cleanup()

    # Load motion encoder (contains BatchNorm - must be FP32 for ONNX export)
    logger.info("Loading MotEncoder...")
    motion_encoder = MotEncoder().to(dtype=export_dtype, device=device).eval()
    motion_encoder_path = weights_path / "motion_encoder.pth"
    if motion_encoder_path.exists():
        state_dict = torch.load(motion_encoder_path, map_location="cpu")
        motion_encoder.load_state_dict(state_dict)
        del state_dict
    motion_encoder.set_attn_processor(AttnProcessor())
    cleanup()

    # Load denoising UNet (explicit reference version for TensorRT)
    logger.info("Loading UNet3DConditionModelExplicit for TensorRT...")
    denoising_unet = UNet3DConditionModelExplicit.from_pretrained_2d(
        str(pretrained_base_path),
        "",
        subfolder="unet",
        unet_additional_kwargs=unet_kwargs,
    ).to(dtype=export_dtype, device=device)

    denoising_unet_path = weights_path / "denoising_unet.pth"
    if denoising_unet_path.exists():
        state_dict = torch.load(denoising_unet_path, map_location="cpu")
        denoising_unet.load_state_dict(state_dict, strict=False)
        del state_dict
        cleanup()

    temporal_module_path = weights_path / "temporal_module.pth"
    if temporal_module_path.exists():
        state_dict = torch.load(temporal_module_path, map_location="cpu")
        denoising_unet.load_state_dict(state_dict, strict=False)
        del state_dict
        cleanup()

    # IMPORTANT: Use standard AttnProcessor for ONNX export
    # xformers is NOT compatible with ONNX tracing!
    denoising_unet.set_attn_processor(AttnProcessor())
    cleanup()

    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(str(vae_path)).to(device=device, dtype=export_dtype)
    vae.set_default_attn_processor()
    cleanup()

    # Setup scheduler (CPU-only, minimal memory)
    scheduler = DDIMScheduler(**sched_kwargs)
    scheduler.to(device)

    # Create timesteps tensor
    # For TensorRT, we use a fixed timestep pattern
    timesteps = torch.tensor(
        [0, 0, 0, 0, 333, 333, 333, 333, 666, 666, 666, 666, 999, 999, 999, 999],
        device=device,
    ).long()
    scheduler.set_step_length(333)

    logger.info("All models loaded. Final memory cleanup...")
    cleanup()
    log_gpu_memory("after loading all models")

    return {
        "pose_guider": pose_guider,
        "motion_encoder": motion_encoder,
        "denoising_unet": denoising_unet,
        "vae": vae,
        "scheduler": scheduler,
        "timesteps": timesteps,
    }


def export_to_onnx(
    models: dict,
    onnx_path: Path,
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Export bundled model to ONNX following official PersonaLive approach.

    Args:
        models: Dictionary of loaded models.
        onnx_path: Path to save ONNX model.
        height: Output height.
        width: Output width.
        batch_size: Batch size.
        device: Target device.
        dtype: Data type for export.
    """
    from .export import export_onnx, optimize_onnx
    from .framed_model import UNetWork

    def cleanup():
        """Aggressive memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    logger.info("Creating bundled UNetWork model...")
    cleanup()

    # Create bundled model
    unet_work = UNetWork(
        pose_guider=models["pose_guider"],
        motion_encoder=models["motion_encoder"],
        denoising_unet=models["denoising_unet"],
        vae=models["vae"],
        scheduler=models["scheduler"],
        timesteps=models["timesteps"],
    )

    # Export to ONNX
    # Note: auto_cast=False because scope's MotEncoder has BatchNorm that
    # doesn't work well with autocast (dtype mismatch issues)
    log_gpu_memory("before ONNX export")
    raw_onnx_path = onnx_path.parent / "unet" / "unet.onnx"
    export_onnx(
        model=unet_work,
        onnx_path=raw_onnx_path,
        opt_image_height=height,
        opt_image_width=width,
        opt_batch_size=batch_size,
        onnx_opset=17,
        dtype=dtype,
        device=device,
        auto_cast=False,  # Disabled due to BatchNorm dtype issues
    )
    # Note: export_onnx handles cleanup of unet_work internally
    cleanup()

    # Clear the models dict
    for key in list(models.keys()):
        del models[key]
    cleanup()
    log_gpu_memory("after ONNX export cleanup")

    # Optimize ONNX (CPU-only, doesn't need GPU)
    logger.info("Optimizing ONNX model...")
    optimize_onnx(raw_onnx_path, onnx_path)
    cleanup()


def main():
    """Main entry point for TensorRT conversion."""
    args = parse_args()

    # Validate inputs
    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    personalive_weights = args.model_dir / "PersonaLive" / "pretrained_weights" / "personalive"
    if not personalive_weights.exists():
        logger.error(f"PersonaLive weights not found at: {personalive_weights}")
        logger.error("Please download PersonaLive models first using: download_models")
        sys.exit(1)

    # Check TensorRT availability
    try:
        from .builder import TRT_AVAILABLE, build_engine, get_engine_path, get_onnx_path

        if not TRT_AVAILABLE:
            raise ImportError()
    except ImportError:
        logger.error("TensorRT is not available.")
        logger.error("Install with: pip install daydream-scope[tensorrt]")
        sys.exit(1)

    # Setup
    device = torch.device(args.device)
    dtype = torch.float32 if args.fp32 else torch.float16

    # ONNX export uses 256x256 to save memory (official approach)
    # TRT engine is built at target resolution with dynamic shapes
    onnx_height, onnx_width = 256, 256

    logger.info(f"Converting PersonaLive models to TensorRT")
    logger.info(f"  Model directory: {args.model_dir}")
    logger.info(f"  Target resolution: {args.height}x{args.width}")
    logger.info(f"  ONNX export resolution: {onnx_height}x{onnx_width} (to save memory)")
    logger.info(f"  Precision: {'FP32' if args.fp32 else 'FP16'}")
    logger.info(f"  Device: {device}")

    start_time = time.time()

    # Get paths
    onnx_path = get_onnx_path(args.model_dir)
    engine_path = get_engine_path(args.model_dir, args.height, args.width)

    # Step 1: Export to ONNX at 256x256 (if needed)
    # This uses less memory - TRT handles target resolution via dynamic shapes
    if not args.skip_onnx or not onnx_path.exists():
        logger.info("Step 1: Exporting to ONNX at 256x256...")
        models = load_models(args.model_dir, device, dtype)
        export_to_onnx(
            models=models,
            onnx_path=onnx_path,
            height=onnx_height,  # 256 to save memory
            width=onnx_width,    # 256 to save memory
            batch_size=args.batch_size,
            device=device,
            dtype=dtype,
        )
        # Clean up models
        del models
        gc.collect()
        torch.cuda.empty_cache()
    else:
        logger.info("Step 1: Skipping ONNX export (using existing files)")

    # Step 2: Build TensorRT engine at target resolution
    logger.info("Step 2: Building TensorRT engine...")
    logger.info(f"  Building for {args.height}x{args.width} resolution")
    logger.info("  This may take 10-30 minutes depending on your GPU.")

    build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        height=args.height,
        width=args.width,
        fp16=not args.fp32,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - start_time
    logger.info(f"Conversion completed in {elapsed / 60:.1f} minutes")
    logger.info(f"TensorRT engine saved to: {engine_path}")
    logger.info("")
    logger.info("The PersonaLive pipeline will automatically use TensorRT")
    logger.info("when the engine file is present.")


if __name__ == "__main__":
    main()
