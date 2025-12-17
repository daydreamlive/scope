"""TensorRT engine builder using polygraphy.

This module provides utilities to build TensorRT engines from ONNX models
using NVIDIA's polygraphy library.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polygraphy.backend.trt import Profile

logger = logging.getLogger(__name__)

# Check for TensorRT availability
TRT_AVAILABLE = False
try:
    import tensorrt as trt
    from polygraphy.backend.trt import (
        CreateConfig,
        Profile,
        engine_from_network,
        network_from_onnx_path,
        save_engine,
    )
    from polygraphy.logger import G_LOGGER

    TRT_AVAILABLE = True
    # Set polygraphy logger to WARNING to reduce noise
    G_LOGGER.severity = G_LOGGER.WARNING
except ImportError:
    pass


def get_engine_path(model_dir: Path, height: int = 512, width: int = 512) -> Path:
    """Get the expected TensorRT engine path for given resolution.

    Args:
        model_dir: Base model directory.
        height: Output height.
        width: Output width.

    Returns:
        Path to the TensorRT engine file.
    """
    tensorrt_dir = model_dir / "PersonaLive" / "pretrained_weights" / "tensorrt"
    return tensorrt_dir / f"unet_work_{height}x{width}.engine"


def get_onnx_path(model_dir: Path) -> Path:
    """Get the ONNX model path.

    Args:
        model_dir: Base model directory.

    Returns:
        Path to the optimized ONNX model.
    """
    onnx_dir = model_dir / "PersonaLive" / "pretrained_weights" / "onnx"
    return onnx_dir / "unet_opt" / "unet_opt.onnx"


def create_optimization_profile(
    batch_size: int = 1,
    height: int = 512,
    width: int = 512,
    temporal_window_size: int = 4,
    temporal_adaptive_step: int = 4,
) -> "Profile":
    """Create TensorRT optimization profile for PersonaLive.

    This defines the input shapes for the TensorRT engine, including
    min/opt/max shapes for dynamic dimensions.

    Args:
        batch_size: Batch size (usually 1).
        height: Output height.
        width: Output width.
        temporal_window_size: Number of frames per temporal window.
        temporal_adaptive_step: Number of adaptive steps.

    Returns:
        Polygraphy Profile object.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available. Install with: pip install daydream-scope[tensorrt]")

    # Derived dimensions
    tb = temporal_window_size * temporal_adaptive_step  # temporal batch size (16)
    lh, lw = height // 8, width // 8  # latent height/width
    ml, mc = 32, 16  # motion latent size, motion channels
    mh, mw = 224, 224  # motion input size
    emb = 768  # CLIP embedding dim
    lc, ic = 4, 3  # latent channels, image channels

    # UNet channel dimensions
    cd0, cd1, cd2, cm = 320, 640, 1280, 1280
    cu1, cu2, cu3 = 1280, 640, 320

    profile = Profile()

    # Fixed shape inputs
    fixed_inputs = {
        "sample": (batch_size, lc, tb, lh, lw),
        "encoder_hidden_states": (batch_size, 1, emb),
        "motion_hidden_states": (batch_size, temporal_window_size * (temporal_adaptive_step - 1), ml, mc),
        "motion": (batch_size, ic, temporal_window_size, mh, mw),
        "pose_cond_fea": (batch_size, cd0, temporal_window_size * (temporal_adaptive_step - 1), lh, lw),
        "pose": (batch_size, ic, temporal_window_size, height, width),
        "new_noise": (batch_size, lc, temporal_window_size, lh, lw),
    }

    for name, shape in fixed_inputs.items():
        profile.add(name, min=shape, opt=shape, max=shape)

    # Dynamic shape inputs (for reference hidden states that can grow with keyframes)
    # Base shapes at 1x, max at 4x for keyframe accumulation
    dynamic_inputs = {
        "d00": (batch_size, lh * lw, cd0),
        "d01": (batch_size, lh * lw, cd0),
        "d10": (batch_size, lh * lw // 4, cd1),
        "d11": (batch_size, lh * lw // 4, cd1),
        "d20": (batch_size, lh * lw // 16, cd2),
        "d21": (batch_size, lh * lw // 16, cd2),
        "m": (batch_size, lh * lw // 64, cm),
        "u10": (batch_size, lh * lw // 16, cu1),
        "u11": (batch_size, lh * lw // 16, cu1),
        "u12": (batch_size, lh * lw // 16, cu1),
        "u20": (batch_size, lh * lw // 4, cu2),
        "u21": (batch_size, lh * lw // 4, cu2),
        "u22": (batch_size, lh * lw // 4, cu2),
        "u30": (batch_size, lh * lw, cu3),
        "u31": (batch_size, lh * lw, cu3),
        "u32": (batch_size, lh * lw, cu3),
    }

    for name, base_shape in dynamic_inputs.items():
        dim0, dim1_base, dim2 = base_shape
        # Allow 1x to 4x scaling for keyframe accumulation
        min_shape = (dim0, dim1_base, dim2)
        opt_shape = (dim0, dim1_base * 2, dim2)
        max_shape = (dim0, dim1_base * 4, dim2)
        profile.add(name, min=min_shape, opt=opt_shape, max=max_shape)

    return profile


def build_engine(
    onnx_path: Path | str,
    engine_path: Path | str,
    height: int = 512,
    width: int = 512,
    fp16: bool = True,
    batch_size: int = 1,
    temporal_window_size: int = 4,
    temporal_adaptive_step: int = 4,
) -> bool:
    """Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Path to save the TensorRT engine.
        height: Output height.
        width: Output width.
        fp16: Use FP16 precision.
        batch_size: Batch size.
        temporal_window_size: Temporal window size.
        temporal_adaptive_step: Temporal adaptive steps.

    Returns:
        True if engine was built successfully.

    Raises:
        RuntimeError: If TensorRT is not available.
        FileNotFoundError: If ONNX model doesn't exist.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available. Install with: pip install daydream-scope[tensorrt]")

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    # Create output directory
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building TensorRT engine from {onnx_path}")
    logger.info(f"Resolution: {height}x{width}, FP16: {fp16}")

    # Create optimization profile
    profile = create_optimization_profile(
        batch_size=batch_size,
        height=height,
        width=width,
        temporal_window_size=temporal_window_size,
        temporal_adaptive_step=temporal_adaptive_step,
    )

    # Build engine using polygraphy
    logger.info("Loading ONNX model and building TensorRT engine...")
    logger.info("This may take 10-30 minutes depending on your GPU.")

    try:
        engine = engine_from_network(
            network_from_onnx_path(
                str(onnx_path),
                flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM],
            ),
            config=CreateConfig(
                fp16=fp16,
                refittable=False,
                profiles=[profile],
            ),
        )

        logger.info(f"Saving engine to {engine_path}")
        save_engine(engine, str(engine_path))

        logger.info("TensorRT engine built successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to build TensorRT engine: {e}")
        raise


def is_engine_available(model_dir: Path, height: int = 512, width: int = 512) -> bool:
    """Check if a TensorRT engine is available for the given configuration.

    Args:
        model_dir: Base model directory.
        height: Output height.
        width: Output width.

    Returns:
        True if engine file exists.
    """
    engine_path = get_engine_path(model_dir, height, width)
    return engine_path.exists()

