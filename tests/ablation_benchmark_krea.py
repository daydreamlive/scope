"""Krea Realtime Video ablation benchmark for arxiv paper.

Measures baseline Krea (14B) vs Krea + VACE across configurations.
Records: per-chunk latency, FPS, peak VRAM. Saves output videos.
"""

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir


@dataclass
class BenchmarkResult:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    fps_values: list[float] = field(default_factory=list)
    peak_vram_mb: float = 0.0
    frames_per_chunk: int = 0

    @property
    def avg_latency_ms(self):
        return (
            sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
        )

    @property
    def avg_fps(self):
        return sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0

    @property
    def peak_fps(self):
        return max(self.fps_values) if self.fps_values else 0


def make_depth_frames(device, dtype, height, width, frames_per_chunk):
    depth = torch.linspace(0, 1, width, device=device, dtype=dtype)
    depth = depth.unsqueeze(0).unsqueeze(0).expand(frames_per_chunk, 3, height, -1)
    depth = depth.unsqueeze(0).permute(0, 2, 1, 3, 4)
    depth = depth * 2.0 - 1.0
    return depth


def make_inpaint_mask(device, dtype, height, width, frames_per_chunk):
    mask = torch.zeros(
        1, 1, frames_per_chunk, height, width, device=device, dtype=dtype
    )
    h_start, h_end = height // 4, 3 * height // 4
    w_start, w_end = width // 4, 3 * width // 4
    mask[:, :, :, h_start:h_end, w_start:w_end] = 1.0
    return mask


def make_inpaint_frames(device, dtype, height, width, frames_per_chunk):
    frames = torch.randn(
        1, 3, frames_per_chunk, height, width, device=device, dtype=dtype
    )
    return frames.clamp(-1, 1)


def save_video(frames_list, output_path, fps=16):
    """Save a list of frame tensors as an mp4 video."""
    import torchvision.io

    all_frames = torch.cat(frames_list, dim=0)
    if all_frames.shape[1] == 3:  # [F, C, H, W]
        all_frames = all_frames.permute(0, 2, 3, 1)
    if all_frames.dtype != torch.uint8:
        all_frames = (all_frames.clamp(0, 1) * 255).to(torch.uint8)
    torchvision.io.write_video(str(output_path), all_frames.cpu(), fps=fps)
    print(f"  Saved video: {output_path}")


def run_krea(
    name,
    vace_path,
    vace_input_frames_fn=None,
    vace_input_masks_fn=None,
    first_frame_image=None,
    prompt="",
    height=256,
    width=448,
    num_chunks=15,
    warmup_chunks=3,
    frames_per_chunk=3,
    vace_context_scale=1.0,
    output_dir=None,
):
    from scope.core.pipelines.krea_realtime_video.pipeline import (
        KreaRealtimeVideoPipeline,
    )
    from scope.core.pipelines.utils import Quantization

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
        / "krea_realtime_video"
    )

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path(
                    "krea-realtime-video/krea-realtime-video-14b.safetensors"
                )
            ),
            "vace_path": vace_path,
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
            "model_config": OmegaConf.load(script_dir / "model.yaml"),
            "height": height,
            "width": width,
            "vae_type": "wan",
        }
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Krea 14B needs FP8 quantization to fit on 32GB VRAM
    pipeline = KreaRealtimeVideoPipeline(
        pipeline_config,
        quantization=Quantization.FP8_E4M3FN,
        compile=False,
        device=device,
        dtype=dtype,
    )
    print(f"  Pipeline loaded. VACE: {'enabled' if vace_path else 'disabled'}")

    result = BenchmarkResult(name=name, frames_per_chunk=frames_per_chunk)
    all_output_frames = []

    for chunk_idx in range(warmup_chunks + num_chunks):
        is_warmup = chunk_idx < warmup_chunks
        is_first = chunk_idx == 0

        kwargs = {
            "prompts": [{"text": prompt, "weight": 100}],
            "init_cache": is_first,
        }

        if vace_path:
            kwargs["vace_context_scale"] = vace_context_scale

        if vace_input_frames_fn and vace_path:
            kwargs["vace_input_frames"] = vace_input_frames_fn(
                device, dtype, height, width, frames_per_chunk
            )
        if vace_input_masks_fn and vace_path:
            kwargs["vace_input_masks"] = vace_input_masks_fn(
                device, dtype, height, width, frames_per_chunk
            )

        # I2V: pass first_frame_image on the first chunk only
        if first_frame_image is not None and vace_path and is_first:
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
            all_output_frames.append(output.clone())

    result.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(
        f"\n  Results: avg {result.avg_latency_ms:.0f}ms, avg {result.avg_fps:.1f} fps, peak {result.peak_fps:.1f} fps"
    )
    print(f"  Peak VRAM: {result.peak_vram_mb:.0f} MB")

    # Save output video
    if output_dir and all_output_frames:
        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("+", "")
            .replace("(", "")
            .replace(")", "")
        )
        video_path = output_dir / f"{safe_name}.mp4"
        try:
            save_video(all_output_frames, video_path)
        except Exception as e:
            print(f"  Warning: Could not save video: {e}")

    del pipeline, all_output_frames
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print("  Krea Realtime Video (14B) Ablation Benchmark")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(
        f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Small resolution to fit 14B model in VRAM
    height, width = 256, 256
    num_chunks = 15
    warmup_chunks = 3
    frames_per_chunk = 3  # Krea uses num_frame_per_block=3

    print(f"  Resolution: {width}x{height}")
    print(f"  Chunks: {warmup_chunks} warmup + {num_chunks} measured")
    print(f"  Frames/chunk: {frames_per_chunk}")

    vace_path = str(
        get_model_file_path("WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors")
    )

    output_dir = Path(__file__).parent / "ablation_krea_output"
    output_dir.mkdir(exist_ok=True)

    results = []

    # 1. Baseline: no VACE
    results.append(
        run_krea(
            "Krea Baseline (no VACE)",
            vace_path=None,
            prompt="a beautiful landscape with mountains and clouds",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
            output_dir=output_dir,
        )
    )

    # 2. VACE + Depth Control
    results.append(
        run_krea(
            "Krea + Depth Control",
            vace_path=vace_path,
            vace_input_frames_fn=make_depth_frames,
            prompt="a beautiful landscape with mountains and clouds",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
            output_dir=output_dir,
        )
    )

    # 3. VACE + Inpainting
    results.append(
        run_krea(
            "Krea + Inpainting",
            vace_path=vace_path,
            vace_input_frames_fn=make_inpaint_frames,
            vace_input_masks_fn=make_inpaint_mask,
            prompt="a fireball",
            height=height,
            width=width,
            num_chunks=num_chunks,
            warmup_chunks=warmup_chunks,
            frames_per_chunk=frames_per_chunk,
            output_dir=output_dir,
        )
    )

    # 4. VACE + Extension (I2V)
    first_frame = "frontend/public/assets/woman1.jpg"
    first_frame_path = Path(__file__).parent.parent / first_frame
    if first_frame_path.exists():
        results.append(
            run_krea(
                "Krea + Extension (I2V)",
                vace_path=vace_path,
                first_frame_image=str(first_frame_path),
                prompt="",
                height=height,
                width=width,
                num_chunks=num_chunks,
                warmup_chunks=warmup_chunks,
                frames_per_chunk=frames_per_chunk,
                output_dir=output_dir,
            )
        )
    else:
        print(f"\n  Skipping extension test: {first_frame_path} not found")

    # Print summary table
    print("\n\n" + "=" * 70)
    print("  KREA 14B ABLATION RESULTS")
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

    print(f"\n  Output videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
