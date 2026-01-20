"""Test script for RAFT TensorRT export and inference.

Run with: uv run python -m scope.core.pipelines.optical_flow.test_tensorrt
"""

import logging

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_raft_tensorrt(use_large_model: bool = True, clean_start: bool = True):
    """Test RAFT ONNX export and TensorRT engine build.

    Args:
        use_large_model: If True, test RAFT Large. If False, test RAFT Small.
        clean_start: If True, delete existing ONNX/engine files first.
    """
    from .download import get_engine_path, get_models_dir, get_onnx_path
    from .engine import (
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        TensorRTEngine,
        build_tensorrt_engine,
        export_raft_to_onnx,
        get_gpu_name,
    )

    model_name = "raft_large" if use_large_model else "raft_small"
    model_size = "Large" if use_large_model else "Small"
    height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
    device = "cuda"

    logger.info("=" * 60)
    logger.info(f"Testing RAFT {model_size} with TensorRT")
    logger.info("=" * 60)

    # Get paths
    models_dir = get_models_dir()
    gpu_name = get_gpu_name()
    onnx_path = get_onnx_path(models_dir, height, width, model_name)
    engine_path = get_engine_path(models_dir, height, width, gpu_name, model_name)

    logger.info(f"Models dir: {models_dir}")
    logger.info(f"ONNX path: {onnx_path}")
    logger.info(f"Engine path: {engine_path}")
    logger.info(f"GPU: {gpu_name}")

    # Clean start if requested
    if clean_start:
        if onnx_path.exists():
            logger.info(f"Deleting existing ONNX: {onnx_path}")
            onnx_path.unlink()
        if engine_path.exists():
            logger.info(f"Deleting existing engine: {engine_path}")
            engine_path.unlink()

    # Step 1: Export to ONNX
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 1: Export RAFT to ONNX")
    logger.info(f"{'=' * 60}")

    if not onnx_path.exists():
        success = export_raft_to_onnx(
            onnx_path=onnx_path,
            height=height,
            width=width,
            device=device,
            use_large_model=use_large_model,
        )
        if not success:
            logger.error("ONNX export failed!")
            return False
        logger.info("ONNX export succeeded!")
    else:
        logger.info(f"ONNX already exists: {onnx_path}")

    # Step 2: Build TensorRT engine
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 2: Build TensorRT engine")
    logger.info(f"{'=' * 60}")

    if not engine_path.exists():
        success = build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            min_height=height,
            min_width=width,
            max_height=height,
            max_width=width,
            fp16=True,
        )
        if not success:
            logger.error("TensorRT engine build failed!")
            return False
        logger.info("TensorRT engine build succeeded!")
    else:
        logger.info(f"Engine already exists: {engine_path}")

    # Step 3: Test inference
    logger.info(f"\n{'=' * 60}")
    logger.info("Step 3: Test TensorRT inference")
    logger.info(f"{'=' * 60}")

    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
        )

        from .engine import apply_raft_transforms

        # Load engine
        engine = TensorRTEngine(engine_path)
        engine.load()
        engine.activate()

        input_shape = (1, 3, height, width)
        engine.allocate_buffers(device=device, input_shape=input_shape)

        # Create test inputs
        frame1 = torch.randn(1, 3, height, width, device=device)
        frame2 = torch.randn(1, 3, height, width, device=device)

        # Apply RAFT transforms
        weights = (
            Raft_Large_Weights.DEFAULT
            if use_large_model
            else Raft_Small_Weights.DEFAULT
        )
        frame1, frame2 = apply_raft_transforms(weights, frame1, frame2)

        # Run inference
        feed_dict = {"frame1": frame1, "frame2": frame2}
        cuda_stream = torch.cuda.current_stream().cuda_stream
        result = engine.infer(feed_dict, cuda_stream)

        flow = result["flow"]
        logger.info("Inference succeeded!")
        logger.info(f"Flow shape: {flow.shape}")
        logger.info(f"Flow dtype: {flow.dtype}")
        logger.info(f"Flow min: {flow.min().item():.4f}, max: {flow.max().item():.4f}")

        del engine
        torch.cuda.empty_cache()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"SUCCESS: RAFT {model_size} + TensorRT works!")
        logger.info(f"{'=' * 60}")
        return True

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test RAFT TensorRT export and inference"
    )
    parser.add_argument(
        "--small", action="store_true", help="Test RAFT Small instead of Large"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Don't delete existing files"
    )
    args = parser.parse_args()

    use_large = not args.small
    clean_start = not args.no_clean

    success = test_raft_tensorrt(use_large_model=use_large, clean_start=clean_start)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
