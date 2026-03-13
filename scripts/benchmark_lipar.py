"""Benchmark masked pruning on LongLive V2V pipeline.

Compares V2V inference with masked pruning disabled vs enabled, measuring:
- Per-chunk wall time (ms)
- Per-stage breakdown (preprocess, VAE encode, denoise, VAE decode)
- Peak and per-chunk VRAM usage (allocated and reserved)
- Compression ratio per chunk (kept/total spatial positions from prune_mask)
- Total throughput (frames/sec)
- Optional output quality comparison (PSNR/SSIM between pruned and non-pruned outputs)

Requires TWO video paths: an input video (content to process) and a mask video
(provides inpainting masks directly as frames). The mask video frames ARE the masks;
no diff computation is performed.

Usage:
    uv run python scripts/benchmark_lipar.py path/to/video.mp4 --mask-video path/to/mask.mp4
    uv run python scripts/benchmark_lipar.py path/to/video.mp4 --mask-video path/to/mask.mp4 --num-chunks 20
    uv run python scripts/benchmark_lipar.py path/to/video.mp4 --mask-video path/to/mask.mp4 --save-frames ./output
"""

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir


def load_video_frames(
    video_path: str | Path, max_frames: int = 0
) -> list[torch.Tensor]:
    """Load a video file into a list of [1, H, W, C] uint8 tensors."""
    import torchvision.io as tvio

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_tensor, _, _ = tvio.read_video(str(video_path), output_format="THWC")
    if max_frames > 0:
        frames_tensor = frames_tensor[:max_frames]

    return [f.unsqueeze(0) for f in frames_tensor]


def load_mask_frames(
    mask_video_path: str | Path, max_frames: int = 0
) -> list[torch.Tensor]:
    """Load a mask video into a list of [H, W] grayscale float tensors in [0, 1].

    Each frame is converted to grayscale by averaging across channels.
    """
    import torchvision.io as tvio

    mask_video_path = Path(mask_video_path)
    if not mask_video_path.exists():
        raise FileNotFoundError(f"Mask video not found: {mask_video_path}")

    frames_tensor, _, _ = tvio.read_video(str(mask_video_path), output_format="THWC")
    if max_frames > 0:
        frames_tensor = frames_tensor[:max_frames]

    masks = []
    for frame in frames_tensor:
        # frame: [H, W, C] uint8 -> grayscale float [0, 1]
        gray = frame.float().mean(dim=-1) / 255.0  # [H, W]
        masks.append(gray)

    return masks


def get_chunk_inpainting_mask(
    mask_frames: list[torch.Tensor],
    chunk_idx: int,
    frames_per_chunk: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Build a [1, 1, H, W] binary inpainting mask for a given chunk.

    Takes the mask video frames corresponding to the chunk range, averages them,
    and thresholds to produce a binary mask.

    Args:
        mask_frames: List of [H, W] grayscale float tensors from the mask video.
        chunk_idx: Which chunk to build the mask for.
        frames_per_chunk: Number of frames per chunk.
        threshold: Grayscale threshold (0-1). Pixels above this are marked as
                   regions to regenerate (1.0 = keep), below as prune (0.0).

    Returns:
        mask: [1, 1, H, W] float tensor. 1.0 = regenerate, 0.0 = prune.
    """
    start = (chunk_idx * frames_per_chunk) % max(1, len(mask_frames) - frames_per_chunk)
    chunk = mask_frames[start : start + frames_per_chunk]
    if not chunk:
        chunk = [mask_frames[-1]]

    avg = torch.stack(chunk).mean(dim=0)  # [H, W]
    binary = (avg > threshold).float()
    return binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


def load_pipeline(config):
    """Load the LongLive pipeline once (expensive)."""
    from scope.core.pipelines.longlive.pipeline import LongLivePipeline

    device = torch.device("cuda")
    pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
    return pipeline


# ---------------------------------------------------------------------------
# Stage-level timing via block monkey-patching
# ---------------------------------------------------------------------------


class StageTimer:
    """Collects per-stage wall times and per-stage VRAM deltas within a single pipeline call."""

    def __init__(self):
        self.times: dict[str, float] = {}
        self.vram_peaks: dict[str, float] = {}  # per-stage peak allocated (MB)

    def reset(self):
        self.times = {}
        self.vram_peaks = {}


def _vram_allocated_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 * 1024)


def _vram_reserved_mb() -> float:
    return torch.cuda.memory_reserved() / (1024 * 1024)


class _TimedBlockWrapper:
    """Wraps a pipeline block to record wall time and VRAM in a StageTimer.

    Python dispatches dunder methods via the type, not the instance, so we
    cannot simply set ``block.__call__ = wrapped``.  Instead we replace the
    block entry in sub_blocks with this thin wrapper whose own ``__call__``
    delegates to the original.
    """

    def __init__(self, block, name: str, timer: StageTimer):
        self._block = block
        self._name = name
        self._timer = timer

    def __call__(self, components, state):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        result = self._block(components, state)
        torch.cuda.synchronize()
        self._timer.times[self._name] = self._timer.times.get(self._name, 0.0) + (
            time.perf_counter() - t0
        )
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        self._timer.vram_peaks[self._name] = max(
            self._timer.vram_peaks.get(self._name, 0.0), peak
        )
        return result

    def __getattr__(self, name):
        return getattr(self._block, name)


def instrument_pipeline(pipeline, timer: StageTimer):
    """Replace every block in the pipeline with a timed wrapper."""
    for name in list(pipeline.blocks.sub_blocks.keys()):
        block = pipeline.blocks.sub_blocks[name]
        pipeline.blocks.sub_blocks[name] = _TimedBlockWrapper(block, name, timer)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    pipeline,
    video_frames: list[torch.Tensor],
    prompt: str,
    num_chunks: int,
    warmup_chunks: int,
    masked_pruning: bool,
    mask_frames: list[torch.Tensor] | None,
    frames_per_chunk: int,
    timer: StageTimer,
    save_dir: Path | None = None,
) -> dict:
    """Run a series of V2V chunks and collect timing and VRAM metrics."""
    all_output_frames = []
    latencies_ms = []
    fps_list = []
    compression_ratios = []
    stage_accum: dict[str, list[float]] = {}
    vram_peak_per_chunk: list[float] = []
    vram_allocated_per_chunk: list[float] = []
    vram_reserved_per_chunk: list[float] = []
    stage_vram_peaks: dict[str, float] = {}
    is_first_call = True
    total_chunks = warmup_chunks + num_chunks

    for chunk_idx in range(total_chunks):
        start_frame = (chunk_idx * frames_per_chunk) % max(
            1, len(video_frames) - frames_per_chunk
        )
        chunk_frames = video_frames[start_frame : start_frame + frames_per_chunk]
        while len(chunk_frames) < frames_per_chunk:
            chunk_frames.append(chunk_frames[-1])

        # Use mask video frames as inpainting mask (only after first chunk, when pruning enabled)
        inpainting_mask = None
        if masked_pruning and mask_frames is not None and chunk_idx > 0:
            inpainting_mask = get_chunk_inpainting_mask(
                mask_frames, chunk_idx, frames_per_chunk
            ).to(device="cuda")

        timer.reset()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        output = pipeline(
            video=chunk_frames,
            prompts=[{"text": prompt, "weight": 100}],
            init_cache=is_first_call,
            inpainting_mask=inpainting_mask,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        chunk_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        chunk_alloc = _vram_allocated_mb()
        chunk_reserved = _vram_reserved_mb()

        is_first_call = False
        out_video = output["video"]
        n_frames_out = out_video.shape[0]

        # Accumulate output frames for saving
        if save_dir is not None:
            video_out = out_video  # [T, C, H, W] or [T, H, W, C]
            if (
                video_out.ndim == 4
                and video_out.shape[1] in (1, 3, 4)
                and video_out.shape[1] < video_out.shape[2]
            ):
                video_out = video_out.permute(0, 2, 3, 1)  # TCHW -> THWC
            if video_out.dtype != torch.uint8:
                video_out = (
                    video_out.clamp(0, 1).mul(255).byte()
                    if video_out.max() <= 1.0
                    else video_out.clamp(0, 255).byte()
                )
            all_output_frames.append(video_out.cpu())

        is_warmup = chunk_idx < warmup_chunks
        label = "warmup" if is_warmup else "measured"

        # Get compression ratio from prune_mask in state
        prune_mask = pipeline.state.get("prune_mask", None)
        ratio = None
        if prune_mask is not None:
            kept = prune_mask.sum().item()
            total = prune_mask.numel()
            ratio = kept / total

        if not is_warmup:
            latencies_ms.append(elapsed * 1000)
            fps_list.append(n_frames_out / elapsed)
            if ratio is not None:
                compression_ratios.append(ratio)
            for sname, stime in timer.times.items():
                stage_accum.setdefault(sname, []).append(stime * 1000)
            vram_peak_per_chunk.append(chunk_peak)
            vram_allocated_per_chunk.append(chunk_alloc)
            vram_reserved_per_chunk.append(chunk_reserved)
            for sname, speak in timer.vram_peaks.items():
                stage_vram_peaks[sname] = max(stage_vram_peaks.get(sname, 0.0), speak)

        ratio_str = f"  compression={ratio:.2%}" if ratio is not None else ""
        print(
            f"  [{label}] chunk {chunk_idx:3d}  "
            f"{elapsed * 1000:7.1f}ms  "
            f"{n_frames_out} frames  "
            f"fps={n_frames_out / elapsed:.2f}"
            f"  peak={chunk_peak:.0f}MB"
            f"{ratio_str}"
        )

    # Write single output video from all accumulated frames
    if save_dir is not None and all_output_frames:
        import torchvision.io as tvio

        save_dir.mkdir(parents=True, exist_ok=True)
        combined = torch.cat(all_output_frames, dim=0)
        tvio.write_video(str(save_dir / "output.mp4"), combined, fps=24)
        print(f"  Saved {combined.shape[0]} frames to {save_dir / 'output.mp4'}")

    results = {
        "latencies_ms": latencies_ms,
        "fps_list": fps_list,
        "compression_ratios": compression_ratios,
        "stage_times": {k: statistics.mean(v) for k, v in stage_accum.items()},
        "vram_peak_per_chunk": vram_peak_per_chunk,
        "vram_allocated_per_chunk": vram_allocated_per_chunk,
        "vram_reserved_per_chunk": vram_reserved_per_chunk,
        "stage_vram_peaks": stage_vram_peaks,
    }

    if latencies_ms:
        results["mean_latency_ms"] = statistics.mean(latencies_ms)
        results["median_latency_ms"] = statistics.median(latencies_ms)
        results["std_latency_ms"] = (
            statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        )
        results["min_latency_ms"] = min(latencies_ms)
        results["max_latency_ms"] = max(latencies_ms)
        results["mean_fps"] = statistics.mean(fps_list)
        results["median_fps"] = statistics.median(fps_list)

    if compression_ratios:
        results["mean_compression"] = statistics.mean(compression_ratios)
        results["median_compression"] = statistics.median(compression_ratios)
        results["min_compression"] = min(compression_ratios)
        results["max_compression"] = max(compression_ratios)

    if vram_peak_per_chunk:
        results["vram_peak_max_mb"] = max(vram_peak_per_chunk)
        results["vram_peak_mean_mb"] = statistics.mean(vram_peak_per_chunk)
        results["vram_alloc_mean_mb"] = statistics.mean(vram_allocated_per_chunk)
        results["vram_reserved_mean_mb"] = statistics.mean(vram_reserved_per_chunk)

    return results


def compute_quality_metrics(baseline_dir: Path, pruned_dir: Path) -> dict | None:
    """Compute PSNR and SSIM between baseline and pruned output videos."""
    try:
        import torchvision.io as tvio
    except ImportError:
        return None

    baseline_path = baseline_dir / "output.mp4"
    pruned_path = pruned_dir / "output.mp4"
    if not baseline_path.exists() or not pruned_path.exists():
        return None

    baseline_frames, _, _ = tvio.read_video(str(baseline_path), output_format="TCHW")
    pruned_frames, _, _ = tvio.read_video(str(pruned_path), output_format="TCHW")

    n = min(len(baseline_frames), len(pruned_frames))
    baseline_frames = baseline_frames[:n].float() / 255.0
    pruned_frames = pruned_frames[:n].float() / 255.0

    # PSNR
    mse = ((baseline_frames - pruned_frames) ** 2).mean()
    psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(float("inf"))

    # SSIM (simplified per-frame mean)
    ssim_values = []
    for i in range(n):
        b = baseline_frames[i]  # [C, H, W]
        p = pruned_frames[i]

        mu_b = b.mean()
        mu_p = p.mean()
        sigma_b_sq = ((b - mu_b) ** 2).mean()
        sigma_p_sq = ((p - mu_p) ** 2).mean()
        sigma_bp = ((b - mu_b) * (p - mu_p)).mean()

        c1 = 0.01**2
        c2 = 0.03**2
        ssim = ((2 * mu_b * mu_p + c1) * (2 * sigma_bp + c2)) / (
            (mu_b**2 + mu_p**2 + c1) * (sigma_b_sq + sigma_p_sq + c2)
        )
        ssim_values.append(ssim.item())

    return {
        "psnr_db": psnr.item(),
        "mean_ssim": statistics.mean(ssim_values),
        "min_ssim": min(ssim_values),
        "max_ssim": max(ssim_values),
    }


def print_summary(label: str, results: dict):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    n = len(results["latencies_ms"])
    print(f"  Chunks measured:     {n}")
    print("  Latency (ms):")
    print(f"    mean:    {results['mean_latency_ms']:>8.1f}")
    print(f"    median:  {results['median_latency_ms']:>8.1f}")
    print(f"    std:     {results['std_latency_ms']:>8.1f}")
    print(f"    min:     {results['min_latency_ms']:>8.1f}")
    print(f"    max:     {results['max_latency_ms']:>8.1f}")
    print("  Throughput (fps):")
    print(f"    mean:    {results['mean_fps']:>8.2f}")
    print(f"    median:  {results['median_fps']:>8.2f}")
    if results.get("compression_ratios"):
        print("  Compression ratio (kept tokens / total):")
        print(f"    mean:    {results['mean_compression']:>8.2%}")
        print(f"    median:  {results['median_compression']:>8.2%}")
        print(f"    min:     {results['min_compression']:>8.2%}")
        print(f"    max:     {results['max_compression']:>8.2%}")
    if results.get("vram_peak_max_mb") is not None:
        print("  VRAM (MB):")
        print(
            f"    peak allocated (max across chunks): {results['vram_peak_max_mb']:>8.0f}"
        )
        print(
            f"    peak allocated (mean per chunk):    {results['vram_peak_mean_mb']:>8.0f}"
        )
        print(
            f"    allocated after chunk (mean):       {results['vram_alloc_mean_mb']:>8.0f}"
        )
        print(
            f"    reserved after chunk (mean):        {results['vram_reserved_mean_mb']:>8.0f}"
        )
    if results.get("stage_vram_peaks"):
        print("  Per-stage peak VRAM (MB):")
        for sname, speak in sorted(
            results["stage_vram_peaks"].items(), key=lambda x: -x[1]
        ):
            print(f"    {sname:<40s} {speak:>8.0f}")
    if results.get("stage_times"):
        print("  Stage breakdown (mean ms):")
        total = sum(results["stage_times"].values())
        for sname, stime in sorted(results["stage_times"].items(), key=lambda x: -x[1]):
            pct = stime / total * 100 if total > 0 else 0
            print(f"    {sname:<40s} {stime:>7.1f}  ({pct:4.1f}%)")
        print(f"    {'TOTAL':<40s} {total:>7.1f}")


def print_comparison(baseline: dict, pruned: dict, quality: dict | None = None):
    print(f"\n{'=' * 60}")
    print("  COMPARISON (PRUNING ON vs OFF)")
    print(f"{'=' * 60}")

    lat_b = baseline["mean_latency_ms"]
    lat_p = pruned["mean_latency_ms"]
    lat_diff_pct = (lat_p - lat_b) / lat_b * 100

    fps_b = baseline["mean_fps"]
    fps_p = pruned["mean_fps"]
    fps_diff_pct = (fps_p - fps_b) / fps_b * 100

    print(f"  Mean latency:   {lat_b:.1f}ms -> {lat_p:.1f}ms  ({lat_diff_pct:+.1f}%)")
    print(f"  Mean fps:       {fps_b:.2f} -> {fps_p:.2f}  ({fps_diff_pct:+.1f}%)")

    if pruned.get("mean_compression") is not None:
        print(f"  Mean compression: {pruned['mean_compression']:.2%} tokens kept")

    print(f"  Speedup: {lat_b / lat_p:.2f}x" if lat_p > 0 else "  Speedup: N/A")

    # VRAM comparison
    vram_b = baseline.get("vram_peak_max_mb")
    vram_p = pruned.get("vram_peak_max_mb")
    if vram_b is not None and vram_p is not None:
        vram_diff = vram_p - vram_b
        vram_pct = vram_diff / vram_b * 100 if vram_b > 0 else 0
        print(
            f"  Peak VRAM:      {vram_b:.0f}MB -> {vram_p:.0f}MB  ({vram_pct:+.1f}%, {vram_diff:+.0f}MB)"
        )
    alloc_b = baseline.get("vram_alloc_mean_mb")
    alloc_p = pruned.get("vram_alloc_mean_mb")
    if alloc_b is not None and alloc_p is not None:
        alloc_diff = alloc_p - alloc_b
        print(
            f"  Steady VRAM:    {alloc_b:.0f}MB -> {alloc_p:.0f}MB  ({alloc_diff:+.0f}MB)"
        )

    # Quality metrics
    if quality is not None:
        print("\n  Output quality (pruned vs baseline):")
        print(f"    PSNR:       {quality['psnr_db']:>8.2f} dB")
        print(f"    SSIM mean:  {quality['mean_ssim']:>8.4f}")
        print(f"    SSIM range: [{quality['min_ssim']:.4f}, {quality['max_ssim']:.4f}]")

    # Per-stage comparison
    b_stages = baseline.get("stage_times", {})
    p_stages = pruned.get("stage_times", {})
    all_stages = sorted(set(b_stages) | set(p_stages))
    if all_stages:
        print("\n  Per-stage comparison (mean ms):")
        print(f"    {'stage':<40s} {'off':>7s}  {'on':>7s}  {'delta':>8s}")
        print(f"    {'-' * 40} {'-' * 7}  {'-' * 7}  {'-' * 8}")
        for s in all_stages:
            bv = b_stages.get(s, 0)
            pv = p_stages.get(s, 0)
            delta = pv - bv
            print(f"    {s:<40s} {bv:>7.1f}  {pv:>7.1f}  {delta:>+7.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark masked pruning on LongLive V2V pipeline"
    )
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=0,
        help="Max frames to decode from video (0 = all)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=10,
        help="Number of measured chunks per config",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=3,
        help="Warmup chunks (excluded from stats)",
    )
    parser.add_argument("--height", type=int, default=512, help="Video height")
    parser.add_argument("--width", type=int, default=512, help="Video width")
    parser.add_argument(
        "--mask-video",
        type=str,
        required=True,
        help="Path to mask video file (frames are used directly as inpainting masks)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A panda sitting in the grass, looking around.",
        help="Prompt text",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline (pruning off) run",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        nargs="+",
        default=[1000, 750],
        help="Denoising step schedule",
    )
    parser.add_argument(
        "--save-frames",
        type=str,
        default=None,
        help="Directory to save output frames (e.g. ./output)",
    )
    args = parser.parse_args()

    frames_per_chunk = 12

    print("Benchmark config:")
    print(f"  Video:            {args.video}")
    print(f"  Mask video:       {args.mask_video}")
    print(f"  Resolution:       {args.width}x{args.height}")
    print(f"  Measured chunks:  {args.num_chunks}")
    print(f"  Warmup chunks:    {args.warmup_chunks}")
    print(f"  Frames per chunk: {frames_per_chunk}")
    print(f"  Denoising steps:  {args.denoising_steps}")
    print(f"  Seed:             {args.seed}")
    print()

    print(f"Loading video from {args.video}...")
    video_frames = load_video_frames(args.video, max_frames=args.max_video_frames)
    h, w = video_frames[0].shape[1], video_frames[0].shape[2]
    print(f"  Loaded {len(video_frames)} frames at {w}x{h}")
    total_chunks_needed = args.warmup_chunks + args.num_chunks
    total_frames_needed = total_chunks_needed * frames_per_chunk
    if len(video_frames) < total_frames_needed:
        print(
            f"  Video has {len(video_frames)} frames, need ~{total_frames_needed} for {total_chunks_needed} chunks; frames will cycle."
        )

    print(f"Loading mask video from {args.mask_video}...")
    mask_frames = load_mask_frames(args.mask_video, max_frames=args.max_video_frames)
    print(
        f"  Loaded {len(mask_frames)} mask frames at {mask_frames[0].shape[1]}x{mask_frames[0].shape[0]}"
    )
    print()

    # Load pipeline with masked pruning enabled so the model wrapper is in place.
    # Pruning is toggled per-run by passing or omitting inpainting_mask.
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
            "model_config": OmegaConf.load(
                Path(__file__).resolve().parent.parent
                / "src"
                / "scope"
                / "core"
                / "pipelines"
                / "longlive"
                / "model.yaml"
            ),
            "height": args.height,
            "width": args.width,
            "base_seed": args.seed,
            "manage_cache": True,
            "denoising_steps": args.denoising_steps,
            "masked_pruning_enabled": True,
        }
    )

    print("Loading LongLive pipeline (with masked pruning wrapper)...")
    t0 = time.perf_counter()
    pipeline = load_pipeline(config)
    print(f"Pipeline loaded in {time.perf_counter() - t0:.1f}s\n")

    # Instrument pipeline blocks for per-stage timing
    timer = StageTimer()
    instrument_pipeline(pipeline, timer)

    save_root = Path(args.save_frames) if args.save_frames else None

    baseline_results = None
    pruned_results = None

    # --- Baseline: masked pruning disabled (no inpainting mask passed) ---
    if not args.skip_baseline:
        print(f"{'=' * 60}")
        print("  BASELINE: MASKED PRUNING DISABLED")
        print(f"{'=' * 60}")

        pipeline.first_call = True
        pipeline.last_mode = None

        baseline_results = run_benchmark(
            pipeline=pipeline,
            video_frames=video_frames,
            prompt=args.prompt,
            num_chunks=args.num_chunks,
            warmup_chunks=args.warmup_chunks,
            masked_pruning=False,
            mask_frames=None,
            frames_per_chunk=frames_per_chunk,
            timer=timer,
            save_dir=save_root / "baseline" if save_root else None,
        )
        print_summary("BASELINE: MASKED PRUNING DISABLED", baseline_results)

        pipeline.first_call = True
        pipeline.last_mode = None
        gc.collect()
        torch.cuda.empty_cache()
        print()

    # --- Masked pruning enabled ---
    print(f"{'=' * 60}")
    print(f"  MASKED PRUNING ENABLED (mask_video={args.mask_video})")
    print(f"{'=' * 60}")

    pipeline.first_call = True
    pipeline.last_mode = None

    pruned_results = run_benchmark(
        pipeline=pipeline,
        video_frames=video_frames,
        prompt=args.prompt,
        num_chunks=args.num_chunks,
        warmup_chunks=args.warmup_chunks,
        masked_pruning=True,
        mask_frames=mask_frames,
        frames_per_chunk=frames_per_chunk,
        timer=timer,
        save_dir=save_root / "pruned" if save_root else None,
    )
    print_summary("MASKED PRUNING ENABLED", pruned_results)

    # --- Comparison ---
    if baseline_results is not None and pruned_results is not None:
        quality = None
        if save_root is not None:
            quality = compute_quality_metrics(
                save_root / "baseline", save_root / "pruned"
            )
        print_comparison(baseline_results, pruned_results, quality)

    print()


if __name__ == "__main__":
    main()
