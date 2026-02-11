"""
Ablation benchmark for arxiv paper.

Measures baseline LongLive vs LongLive + VACE across configurations.
Records: per-chunk latency, FPS, peak VRAM.
"""

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir


def get_artifact_path(artifact, file_index=0) -> str:
    """Resolve an artifact to its local file path using the same logic as Scope."""
    repo_name = artifact.repo_id.split("/")[-1]
    return str(get_model_file_path(f"{repo_name}/{artifact.files[file_index]}"))


@dataclass
class BenchmarkResult:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    fps_values: list[float] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    frames_per_chunk: int = 12

    @property
    def avg_latency_ms(self) -> float:
        return (
            sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
        )

    @property
    def avg_fps(self) -> float:
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0

    @property
    def peak_fps(self) -> float:
        return max(self.fps_values) if self.fps_values else 0


def run_benchmark(
    name: str,
    vace_path: str | None,
    vace_in_dim: int | None,
    vace_input_frames_fn=None,
    vace_input_masks_fn=None,
    extension_mode: str | None = None,
    first_frame_image: str | None = None,
    prompt: str = "",
    height: int = 368,
    width: int = 640,
    num_chunks: int = 5,
    frames_per_chunk: int = 12,
    warmup_chunks: int = 2,
    vace_context_scale: float = 1.0,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    from scope.core.pipelines.longlive.pipeline import LongLivePipeline

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    script_dir = (
        Path(__file__).parent.parent
        / "src"
        / "scope"
        / "core"
        / "pipelines"
        / "longlive"
    )

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "vace_path": vace_path,
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(script_dir / "model.yaml"),
            "height": height,
            "width": width,
            "vae_type": "tae",
        }
    )

    if vace_in_dim is not None:
        pipeline_config.model_config.base_model_kwargs = (
            pipeline_config.model_config.base_model_kwargs or {}
        )
        pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = vace_in_dim

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=dtype)
    print(f"  Pipeline loaded. VACE: {'enabled' if vace_path else 'disabled'}")

    result = BenchmarkResult(name=name, frames_per_chunk=frames_per_chunk)
    total_chunks = warmup_chunks + num_chunks

    for chunk_idx in range(total_chunks):
        is_warmup = chunk_idx < warmup_chunks
        is_first = chunk_idx == 0

        kwargs = {
            "prompts": [{"text": prompt, "weight": 100}],
        }

        if vace_path:
            kwargs["vace_context_scale"] = vace_context_scale

        # Add VACE inputs if provided
        if vace_input_frames_fn is not None and vace_path:
            kwargs["vace_input_frames"] = vace_input_frames_fn(
                device, dtype, height, width, frames_per_chunk
            )

        if vace_input_masks_fn is not None and vace_path:
            kwargs["vace_input_masks"] = vace_input_masks_fn(
                device, dtype, height, width, frames_per_chunk
            )

        if extension_mode is not None and vace_path and is_first:
            kwargs["extension_mode"] = extension_mode
            if first_frame_image:
                kwargs["first_frame_image"] = first_frame_image

        torch.cuda.synchronize()
        start = time.perf_counter()

        result_dict = pipeline(**kwargs)
        output = result_dict["video"] if isinstance(result_dict, dict) else result_dict

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        num_frames = output.shape[0]
        fps = (num_frames / elapsed_ms) * 1000

        label = "WARMUP" if is_warmup else "BENCH"
        print(
            f"  [{label}] Chunk {chunk_idx}: {elapsed_ms:.0f}ms, {fps:.1f} fps ({num_frames} frames)"
        )

        if not is_warmup:
            result.latencies_ms.append(elapsed_ms)
            result.fps_values.append(fps)

    result.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(
        f"\n  Results: avg {result.avg_latency_ms:.0f}ms, avg {result.avg_fps:.1f} fps, peak {result.peak_fps:.1f} fps"
    )
    print(f"  Peak VRAM: {result.peak_vram_mb:.0f} MB")

    # Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return result


def make_depth_frames(device, dtype, height, width, frames_per_chunk):
    """Create synthetic depth map input."""
    # Simple gradient depth map
    depth = torch.linspace(0, 1, width, device=device, dtype=dtype)
    depth = depth.unsqueeze(0).unsqueeze(0).expand(frames_per_chunk, 3, height, -1)
    depth = depth.unsqueeze(0)  # [1, frames, 3, H, W] -> need [1, 3, F, H, W]
    depth = depth.permute(0, 2, 1, 3, 4)
    depth = depth * 2.0 - 1.0  # normalize to [-1, 1]
    return depth


def make_inpaint_mask(device, dtype, height, width, frames_per_chunk):
    """Create a center-rectangle inpainting mask."""
    mask = torch.zeros(
        1, 1, frames_per_chunk, height, width, device=device, dtype=dtype
    )
    h_start, h_end = height // 4, 3 * height // 4
    w_start, w_end = width // 4, 3 * width // 4
    mask[:, :, :, h_start:h_end, w_start:w_end] = 1.0
    return mask


def make_inpaint_frames(device, dtype, height, width, frames_per_chunk):
    """Create synthetic input video for inpainting."""
    frames = torch.randn(
        1, 3, frames_per_chunk, height, width, device=device, dtype=dtype
    )
    frames = frames.clamp(-1, 1)
    return frames


def make_synthetic_first_frame(height, width):
    """Create a synthetic first frame image and return its path as a temp file."""
    import tempfile

    from PIL import Image

    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                int(255 * x / width),
                int(255 * y / height),
                128,
            )
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def main():
    print("=" * 70)
    print("  VACE Ablation Benchmark for arXiv Paper")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(
        f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    height, width = 368, 640
    num_chunks = 15
    warmup_chunks = 3
    frames_per_chunk = 12

    print(f"  Resolution: {width}x{height}")
    print(f"  Chunks: {warmup_chunks} warmup + {num_chunks} measured")
    print(f"  Frames/chunk: {frames_per_chunk}")

    from scope.core.pipelines.common_artifacts import VACE_ARTIFACT

    vace_path = get_artifact_path(VACE_ARTIFACT)

    results = []

    # 1. Baseline: no VACE
    results.append(
        run_benchmark(
            name="LongLive Baseline (no VACE)",
            vace_path=None,
            vace_in_dim=None,
            prompt="a beautiful landscape with mountains and clouds",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
        )
    )

    # 2. VACE + Depth Control
    results.append(
        run_benchmark(
            name="LongLive + Depth Control",
            vace_path=vace_path,
            vace_in_dim=96,
            vace_input_frames_fn=make_depth_frames,
            prompt="a beautiful landscape with mountains and clouds",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
        )
    )

    # 3. VACE + Inpainting
    results.append(
        run_benchmark(
            name="LongLive + Inpainting",
            vace_path=vace_path,
            vace_in_dim=96,
            vace_input_frames_fn=make_inpaint_frames,
            vace_input_masks_fn=make_inpaint_mask,
            prompt="a fireball",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
        )
    )

    # 4. VACE + Extension (firstframe) - synthetic test image
    first_frame_path = make_synthetic_first_frame(height, width)
    results.append(
        run_benchmark(
            name="LongLive + Extension (I2V)",
            vace_path=vace_path,
            vace_in_dim=96,
            extension_mode="firstframe",
            first_frame_image=first_frame_path,
            prompt="",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
        )
    )

    # Print summary table
    print("\n\n" + "=" * 70)
    print("  ABLATION RESULTS")
    print("=" * 70)
    print(
        f"  {'Configuration':<35} {'Avg Latency':>12} {'Avg FPS':>10} {'Peak FPS':>10} {'Peak VRAM':>12}"
    )
    print(f"  {'-' * 35} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 12}")
    for r in results:
        print(
            f"  {r.name:<35} {r.avg_latency_ms:>9.0f} ms {r.avg_fps:>9.1f} {r.peak_fps:>9.1f} {r.peak_vram_mb:>9.0f} MB"
        )
    print("=" * 70)

    # Print VACE overhead
    if len(results) >= 2:
        baseline = results[0]
        print("\n  VACE Overhead vs Baseline:")
        for r in results[1:]:
            overhead_ms = r.avg_latency_ms - baseline.avg_latency_ms
            overhead_pct = (
                (overhead_ms / baseline.avg_latency_ms) * 100
                if baseline.avg_latency_ms > 0
                else 0
            )
            vram_delta = r.peak_vram_mb - baseline.peak_vram_mb
            print(
                f"    {r.name}: +{overhead_ms:.0f}ms (+{overhead_pct:.1f}%), +{vram_delta:.0f} MB VRAM"
            )


if __name__ == "__main__":
    main()
