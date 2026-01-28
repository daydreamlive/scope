"""Benchmark script for optical flow pipeline with and without TensorRT.

Run with: uv run python -m scope.core.pipelines.optical_flow.benchmark
"""

import argparse
import logging
import time
from dataclasses import dataclass

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    backend: str
    model_size: str
    num_frames: int
    total_time_s: float
    warmup_time_s: float
    avg_time_per_frame_ms: float
    fps: float
    peak_memory_mb: float


def create_test_video(
    num_frames: int, height: int, width: int, device: torch.device
) -> list[torch.Tensor]:
    """Create synthetic test video frames with motion.

    Creates frames with a moving gradient pattern to ensure
    optical flow has actual motion to detect.
    """
    frames = []
    for i in range(num_frames):
        # Create a frame with a moving diagonal gradient
        x = torch.linspace(0, 1, width, device=device)
        y = torch.linspace(0, 1, height, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="xy")

        # Add motion by shifting the pattern
        offset = i / num_frames
        pattern = ((xx + yy + offset) % 1.0 * 255).to(torch.uint8)

        # Create RGB frame (1, H, W, C) format
        frame = pattern.unsqueeze(-1).expand(-1, -1, 3).unsqueeze(0)
        frames.append(frame)

    return frames


def run_benchmark(
    use_tensorrt: bool,
    model_size: str,
    num_frames: int,
    height: int,
    width: int,
    warmup_frames: int,
    device: torch.device,
    profile: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    from omegaconf import OmegaConf

    from .pipeline import OpticalFlowPipeline

    backend = "TensorRT" if use_tensorrt else "PyTorch"
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Benchmarking: {backend} backend, RAFT {model_size}")
    logger.info(f"Frames: {num_frames}, Resolution: {height}x{width}")
    logger.info(f"{'=' * 60}")

    # Create config
    config = OmegaConf.create(
        {
            "model_size": model_size,
            "use_tensorrt": use_tensorrt,
        }
    )

    # Reset CUDA memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    init_start = time.perf_counter()
    pipeline = OpticalFlowPipeline(config, device=device)
    init_time = time.perf_counter() - init_start
    logger.info(f"Pipeline initialized in {init_time:.3f}s")

    # Create test video
    logger.info("Creating test video...")
    video = create_test_video(num_frames, height, width, device)

    # Warmup run
    logger.info(f"Running warmup ({warmup_frames} frames)...")
    warmup_video = video[:warmup_frames]
    warmup_start = time.perf_counter()
    with torch.no_grad():
        _ = pipeline(video=warmup_video)
    if device.type == "cuda":
        torch.cuda.synchronize()
    warmup_time = time.perf_counter() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.3f}s")

    # Benchmark run
    logger.info(f"Running benchmark ({num_frames} frames)...")
    if device.type == "cuda":
        torch.cuda.synchronize()

    if profile:
        from torch.profiler import ProfilerActivity
        from torch.profiler import profile as torch_profile

        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            start = time.perf_counter()
            with torch.no_grad():
                result = pipeline(video=video)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time = time.perf_counter() - start

        # Print profiler results
        print("\n" + "=" * 80)
        print("PROFILER RESULTS (Top 20 by CUDA time)")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20, top_level_events_only=False
            )
        )
    else:
        start = time.perf_counter()
        with torch.no_grad():
            result = pipeline(video=video)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start

    # Calculate metrics
    avg_time_per_frame_ms = (total_time / num_frames) * 1000
    fps = num_frames / total_time

    # Get peak memory
    peak_memory_mb = 0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Verify output
    output_video = result["video"]
    logger.info(f"Output shape: {output_video.shape}")
    logger.info(f"Output dtype: {output_video.dtype}")
    logger.info(f"Output range: [{output_video.min():.3f}, {output_video.max():.3f}]")

    # Log results
    logger.info(f"\nResults for {backend} ({model_size}):")
    logger.info(f"  Total time: {total_time:.3f}s")
    logger.info(f"  Avg per frame: {avg_time_per_frame_ms:.2f}ms")
    logger.info(f"  Throughput: {fps:.2f} FPS")
    if peak_memory_mb > 0:
        logger.info(f"  Peak memory: {peak_memory_mb:.1f} MB")

    # Cleanup
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchmarkResult(
        backend=backend,
        model_size=model_size,
        num_frames=num_frames,
        total_time_s=total_time,
        warmup_time_s=warmup_time,
        avg_time_per_frame_ms=avg_time_per_frame_ms,
        fps=fps,
        peak_memory_mb=peak_memory_mb,
    )


def print_comparison(results: list[BenchmarkResult]):
    """Print a comparison table of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(
        f"{'Backend':<12} {'Model':<8} {'Frames':<8} {'Time (s)':<10} "
        f"{'ms/frame':<10} {'FPS':<10} {'Memory (MB)':<12}"
    )
    print("-" * 80)

    # Results
    for r in results:
        print(
            f"{r.backend:<12} {r.model_size:<8} {r.num_frames:<8} "
            f"{r.total_time_s:<10.3f} {r.avg_time_per_frame_ms:<10.2f} "
            f"{r.fps:<10.2f} {r.peak_memory_mb:<12.1f}"
        )

    print("-" * 80)

    # Speedup calculation if we have both TensorRT and PyTorch results
    pytorch_results = [r for r in results if r.backend == "PyTorch"]
    tensorrt_results = [r for r in results if r.backend == "TensorRT"]

    if pytorch_results and tensorrt_results:
        for pt in pytorch_results:
            for trt in tensorrt_results:
                if pt.model_size == trt.model_size:
                    speedup = pt.total_time_s / trt.total_time_s
                    print(
                        f"\nSpeedup ({trt.model_size}): TensorRT is {speedup:.2f}x faster than PyTorch"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optical flow pipeline with and without TensorRT"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to process (default: 30)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Frame height (default: 512)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Frame width (default: 512)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup frames (default: 5)",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "large", "both"],
        default="small",
        help="Model size to benchmark (default: small)",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="Only benchmark PyTorch backend",
    )
    parser.add_argument(
        "--tensorrt-only",
        action="store_true",
        help="Only benchmark TensorRT backend",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per configuration (default: 1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with torch profiler and print detailed breakdown",
    )
    args = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        logger.error("CUDA is required for benchmarking")
        return 1

    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")

    # Check TensorRT availability
    try:
        import tensorrt  # noqa: F401
        from polygraphy.backend.trt import engine_from_bytes  # noqa: F401

        tensorrt_available = True
        logger.info("TensorRT is available")
    except ImportError:
        tensorrt_available = False
        logger.warning("TensorRT not available, will only benchmark PyTorch")
        if args.tensorrt_only:
            logger.error("--tensorrt-only specified but TensorRT is not available")
            return 1

    # Determine configurations to run
    model_sizes = ["small", "large"] if args.model_size == "both" else [args.model_size]

    backends = []
    if not args.tensorrt_only:
        backends.append(False)  # PyTorch
    if not args.pytorch_only and tensorrt_available:
        backends.append(True)  # TensorRT

    if not backends:
        logger.error("No backends to benchmark")
        return 1

    # Run benchmarks
    results = []
    for model_size in model_sizes:
        for use_tensorrt in backends:
            for iteration in range(args.iterations):
                if args.iterations > 1:
                    logger.info(f"\nIteration {iteration + 1}/{args.iterations}")

                result = run_benchmark(
                    use_tensorrt=use_tensorrt,
                    model_size=model_size,
                    num_frames=args.frames,
                    height=args.height,
                    width=args.width,
                    warmup_frames=args.warmup,
                    device=device,
                    profile=args.profile,
                )
                results.append(result)

    # Print comparison
    print_comparison(results)

    return 0


if __name__ == "__main__":
    exit(main())
