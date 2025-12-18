"""ONNX export utilities for PersonaLive models.

Adapted from PersonaLive official implementation:
PersonaLive/src/modeling/onnx_export.py

Based on: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py
"""

import gc
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@contextmanager
def auto_cast_manager(enabled: bool):
    """Context manager for autocast during ONNX export."""
    if enabled:
        with torch.inference_mode(), torch.autocast("cuda"):
            yield
    else:
        yield


@torch.no_grad()
def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path | str,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
    dtype: torch.dtype,
    device: torch.device,
    auto_cast: bool = True,
) -> None:
    """Export a PyTorch model to ONNX format.

    This follows the official PersonaLive export approach.

    Args:
        model: PyTorch model with get_sample_input, get_input_names,
               get_output_names, get_dynamic_axes methods.
        onnx_path: Path to save the ONNX model.
        opt_image_height: Optimization image height.
        opt_image_width: Optimization image width.
        opt_batch_size: Optimization batch size.
        onnx_opset: ONNX opset version.
        dtype: Data type for sample inputs.
        device: Target device.
        auto_cast: Whether to use autocast during export.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to ONNX: {onnx_path}")
    logger.info(f"  Resolution: {opt_image_height}x{opt_image_width}")
    logger.info(f"  Batch size: {opt_batch_size}")
    logger.info(f"  Opset: {onnx_opset}")
    logger.info(f"  Auto cast: {auto_cast}")

    with auto_cast_manager(auto_cast):
        # Generate sample inputs
        inputs = model.get_sample_input(
            opt_batch_size, opt_image_height, opt_image_width, dtype, device
        )

        logger.info(f"Output names: {model.get_output_names()}")

        # Export to ONNX - use torch.onnx.utils.export like official impl
        torch.onnx.utils.export(
            model,
            inputs,  # Pass dict directly, not as tuple
            str(onnx_path),
            export_params=True,
            opset_version=onnx_opset,
            do_constant_folding=True,
            input_names=model.get_input_names(),
            output_names=model.get_output_names(),
            dynamic_axes=model.get_dynamic_axes(),
        )

    logger.info(f"ONNX model exported to {onnx_path}")

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()


def optimize_onnx(
    onnx_path: Path | str,
    onnx_opt_path: Path | str,
) -> None:
    """Optimize ONNX model and save with external data.

    For large models, this saves weights as external data to avoid
    protobuf size limits.

    Args:
        onnx_path: Path to input ONNX model.
        onnx_opt_path: Path to save optimized ONNX model.
    """
    try:
        import onnx
    except ImportError:
        raise RuntimeError("onnx package required. Install with: pip install onnx")

    onnx_path = Path(onnx_path)
    onnx_opt_path = Path(onnx_opt_path)
    onnx_opt_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving ONNX model with external data: {onnx_opt_path}")

    model = onnx.load(str(onnx_path))
    name = onnx_opt_path.stem

    # Save with external data for large models
    onnx.save(
        model,
        str(onnx_opt_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{name}.onnx.data",
        size_threshold=1024,
    )

    logger.info("ONNX optimization done.")
