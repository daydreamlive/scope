"""ONNX export utilities for PersonaLive models.

This module provides utilities to export PersonaLive models to ONNX format
for subsequent TensorRT conversion.
"""

import gc
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def export_onnx(
    model: torch.nn.Module,
    onnx_path: Path | str,
    sample_inputs: dict[str, torch.Tensor],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> None:
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export.
        onnx_path: Path to save the ONNX model.
        sample_inputs: Sample inputs for tracing.
        input_names: Names of input tensors.
        output_names: Names of output tensors.
        dynamic_axes: Dynamic axis specifications.
        opset_version: ONNX opset version.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting model to ONNX: {onnx_path}")

    # Prepare inputs as tuple in correct order
    inputs = tuple(sample_inputs[name] for name in input_names)

    with torch.inference_mode(), torch.autocast("cuda"):
        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
        )

    logger.info(f"ONNX model exported to {onnx_path}")

    # Clean up
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

    logger.info(f"Optimizing ONNX model: {onnx_path} -> {onnx_opt_path}")

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

    logger.info(f"Optimized ONNX model saved to {onnx_opt_path}")


def verify_onnx(onnx_path: Path | str) -> bool:
    """Verify ONNX model is valid.

    Args:
        onnx_path: Path to ONNX model.

    Returns:
        True if model is valid.
    """
    try:
        import onnx
    except ImportError:
        logger.warning("onnx package not available, skipping verification")
        return True

    onnx_path = Path(onnx_path)

    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX model {onnx_path} is valid")
        return True
    except Exception as e:
        logger.error(f"ONNX model verification failed: {e}")
        return False
