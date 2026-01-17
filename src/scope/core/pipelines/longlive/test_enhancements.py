"""
Test script for FreSca enhancements in LongLive pipeline.

Generates comparison videos with baseline on left, enhanced on right.

Usage:
    uv run python -m scope.core.pipelines.longlive.test_enhancements
    uv run python -m scope.core.pipelines.longlive.test_enhancements --fresca-only
    uv run python -m scope.core.pipelines.longlive.test_enhancements --normalized-only
    uv run python -m scope.core.pipelines.longlive.test_enhancements --all-variants

The default test compares normalized FreSca (with tau-based self-limiting)
against the baseline. Normalized FreSca prevents accumulation over long
generations by clamping the enhancement to a maximum norm ratio (tau).
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .enhanced_pipeline import EnhancedLongLivePipeline


def generate_video_frames(
    pipeline: EnhancedLongLivePipeline,
    prompt_text: str,
    max_chunks: int = 6,
    enhancement_kwargs: dict | None = None,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """Generate video frames with specified enhancement settings.

    Returns:
        Tuple of (frames_tensor, latency_measures, fps_measures)
    """
    outputs = []
    latency_measures = []
    fps_measures = []

    enhancement_kwargs = enhancement_kwargs or {}

    for chunk_idx in range(max_chunks):
        start = time.time()

        prompts = [{"text": prompt_text, "weight": 100}]
        output = pipeline(
            prompts=prompts,
            init_cache=(chunk_idx == 0),
            **enhancement_kwargs,
        )

        num_output_frames = output.shape[0]
        latency = time.time() - start
        fps = num_output_frames / latency

        print(
            f"    Chunk {chunk_idx + 1}/{max_chunks}: "
            f"{num_output_frames} frames, latency={latency:.2f}s, fps={fps:.1f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs.append(output.detach().cpu())

    frames = torch.concat(outputs)
    return frames, latency_measures, fps_measures


def concatenate_side_by_side(
    left_frames: torch.Tensor, right_frames: torch.Tensor
) -> torch.Tensor:
    """Concatenate two videos side by side (left | right).

    Args:
        left_frames: Tensor of shape [T, H, W, C]
        right_frames: Tensor of shape [T, H, W, C]

    Returns:
        Combined tensor of shape [T, H, W*2, C]
    """
    # Ensure same number of frames
    min_frames = min(left_frames.shape[0], right_frames.shape[0])
    left_frames = left_frames[:min_frames]
    right_frames = right_frames[:min_frames]

    # Concatenate along width dimension
    combined = torch.cat([left_frames, right_frames], dim=2)
    return combined


def print_statistics(
    name: str, latency_measures: list[float], fps_measures: list[float]
) -> None:
    """Print performance statistics."""
    if not latency_measures:
        return

    # Skip first chunk (warmup)
    if len(latency_measures) > 1:
        latency_measures = latency_measures[1:]
        fps_measures = fps_measures[1:]

    avg_latency = sum(latency_measures) / len(latency_measures)
    avg_fps = sum(fps_measures) / len(fps_measures)
    print(f"  {name}: avg latency={avg_latency:.2f}s, avg FPS={avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test FreSca and TSR enhancements for LongLive"
    )
    parser.add_argument(
        "--fresca-only",
        action="store_true",
        help="Test unbounded FreSca enhancement only (no tau limit)",
    )
    parser.add_argument(
        "--normalized-only",
        action="store_true",
        help="Test normalized FreSca only (with tau-based self-limiting)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate all variants for comparison (fresca, normalized, both tau values)",
    )
    parser.add_argument(
        "--fresca-scale-high",
        type=float,
        default=1.15,
        help="FreSca high-frequency scaling factor (default: 1.15)",
    )
    parser.add_argument(
        "--fresca-freq-cutoff",
        type=int,
        default=20,
        help="FreSca frequency cutoff radius (default: 20)",
    )
    parser.add_argument(
        "--fresca-tau",
        type=float,
        default=1.2,
        help="Normalized FreSca tau - max norm ratio (default: 1.2). "
        "This bounds the enhancement to prevent accumulation over time.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Maximum chunks to generate (default: 20). Use higher values to test "
        "long-generation behavior and accumulation effects.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cinematic shot of a golden retriever running through a meadow "
        "of wildflowers at sunset, with soft bokeh and warm golden light.",
        help="Prompt for video generation",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (default: 832)",
    )
    args = parser.parse_args()

    # Setup config and pipeline
    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
            "height": args.height,
            "width": args.width,
        }
    )

    device = torch.device("cuda")
    print("Loading EnhancedLongLivePipeline...")
    pipeline = EnhancedLongLivePipeline(config, device=device, dtype=torch.bfloat16)

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "enhancements"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define enhancement variants (excluding baseline which is always generated)
    if args.all_variants:
        variants = [
            (
                "fresca_unbounded",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    # No tau = unbounded, can accumulate
                },
            ),
            (
                f"fresca_normalized_tau{args.fresca_tau}",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    "fresca_tau": args.fresca_tau,
                },
            ),
            (
                "fresca_normalized_tau1.1",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    "fresca_tau": 1.1,
                },
            ),
            (
                "fresca_normalized_tau1.3",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    "fresca_tau": 1.3,
                },
            ),
        ]
    elif args.fresca_only:
        # Test unbounded FreSca (original, can accumulate)
        variants = [
            (
                "fresca_unbounded",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                },
            ),
        ]
    elif args.normalized_only:
        # Test normalized FreSca only
        variants = [
            (
                f"fresca_normalized_tau{args.fresca_tau}",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    "fresca_tau": args.fresca_tau,
                },
            ),
        ]
    else:
        # Default: test normalized FreSca (recommended for long generations)
        variants = [
            (
                f"fresca_normalized_tau{args.fresca_tau}",
                {
                    "enable_fresca": True,
                    "fresca_scale_high": args.fresca_scale_high,
                    "fresca_freq_cutoff": args.fresca_freq_cutoff,
                    "fresca_tau": args.fresca_tau,
                },
            ),
        ]

    print(f"\nPrompt: {args.prompt}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Max chunks: {args.max_chunks}")
    print(f"Variants to compare: {[v[0] for v in variants]}\n")

    # Generate baseline first
    print("=" * 60)
    print("Generating: baseline (no enhancements)")
    print("=" * 60)

    baseline_frames, baseline_latency, baseline_fps = generate_video_frames(
        pipeline,
        args.prompt,
        max_chunks=args.max_chunks,
        enhancement_kwargs={},
    )
    print(f"  Baseline frames: {baseline_frames.shape[0]}")

    # Save standalone baseline
    baseline_path = output_dir / "baseline.mp4"
    export_to_video(baseline_frames.contiguous().numpy(), baseline_path, fps=16)
    print(f"  Saved baseline to: {baseline_path}")

    results = {"baseline": {"latency": baseline_latency, "fps": baseline_fps}}

    # Generate each variant and create comparison video
    for variant_name, enhancement_kwargs in variants:
        print(f"\n{'=' * 60}")
        print(f"Generating: {variant_name}")
        print(f"Settings: {enhancement_kwargs}")
        print("=" * 60)

        variant_frames, variant_latency, variant_fps = generate_video_frames(
            pipeline,
            args.prompt,
            max_chunks=args.max_chunks,
            enhancement_kwargs=enhancement_kwargs,
        )
        print(f"  {variant_name} frames: {variant_frames.shape[0]}")

        results[variant_name] = {"latency": variant_latency, "fps": variant_fps}

        # Save standalone variant
        variant_path = output_dir / f"{variant_name}.mp4"
        export_to_video(variant_frames.contiguous().numpy(), variant_path, fps=16)
        print(f"  Saved {variant_name} to: {variant_path}")

        # Create side-by-side comparison (baseline | variant)
        comparison_frames = concatenate_side_by_side(baseline_frames, variant_frames)
        comparison_path = output_dir / f"compare_baseline_vs_{variant_name}.mp4"
        export_to_video(comparison_frames.contiguous().numpy(), comparison_path, fps=16)
        print(f"  Saved comparison to: {comparison_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    for name, data in results.items():
        print_statistics(name, data["latency"], data["fps"])

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print("  baseline.mp4 - No enhancements")
    for variant_name, _ in variants:
        print(f"  {variant_name}.mp4 - {variant_name} only")
        print(f"  compare_baseline_vs_{variant_name}.mp4 - Side-by-side comparison")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
