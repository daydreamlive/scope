"""
Benchmark script for VACE torch.compile optimizations.

Tests both isolated VACE components and end-to-end pipeline performance.
Focused on continual conditioning (depth maps) and inpainting tasks.

Usage:
    python -m scope.core.pipelines.longlive.benchmark_vace
    python -m scope.core.pipelines.longlive.benchmark_vace --mode depth
    python -m scope.core.pipelines.longlive.benchmark_vace --mode inpainting
    python -m scope.core.pipelines.longlive.benchmark_vace --mode both
    python -m scope.core.pipelines.longlive.benchmark_vace --compile  # Enable torch.compile
"""

import argparse
import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline


@dataclass
class TimingResult:
    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self):
        return sum(self.times_ms) / len(self.times_ms) if self.times_ms else 0

    @property
    def min_ms(self):
        return min(self.times_ms) if self.times_ms else 0

    @property
    def max_ms(self):
        return max(self.times_ms) if self.times_ms else 0

    @property
    def std_ms(self):
        if len(self.times_ms) < 2:
            return 0
        mean = self.mean_ms
        return (
            sum((t - mean) ** 2 for t in self.times_ms) / (len(self.times_ms) - 1)
        ) ** 0.5


@contextmanager
def cuda_timer(timing_result: TimingResult):
    """Context manager for GPU-accurate timing using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    timing_result.times_ms.append(start.elapsed_time(end))


class VACEBenchmark:
    def __init__(
        self,
        use_compile=False,
        use_fp8=False,
        height=512,
        width=512,
        num_chunks=3,
        frames_per_chunk=12,
    ):
        self.use_compile = use_compile
        self.use_fp8 = use_fp8
        self.height = height
        self.width = width
        self.num_chunks = num_chunks
        self.frames_per_chunk = frames_per_chunk
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        # Timing storage
        self.timings: dict[str, TimingResult] = {}

        self._init_pipeline()

    def _get_timing(self, name: str) -> TimingResult:
        if name not in self.timings:
            self.timings[name] = TimingResult(name)
        return self.timings[name]

    def _init_pipeline(self):
        print("=" * 80)
        print(f"  VACE Benchmark (compile={'ON' if self.use_compile else 'OFF'})")
        print("=" * 80)

        vace_path = str(
            get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
        )

        script_dir = Path(__file__).parent
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
                "height": self.height,
                "width": self.width,
                "vae_type": "tae",
            }
        )

        # Set vace_in_dim for depth/inpainting modes
        pipeline_config.model_config.base_model_kwargs = (
            pipeline_config.model_config.base_model_kwargs or {}
        )
        pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

        from scope.core.pipelines.enums import Quantization

        quantization = Quantization.FP8_E4M3FN if self.use_fp8 else None

        print(f"Initializing pipeline... (fp8={'ON' if self.use_fp8 else 'OFF'})")
        self.pipeline = LongLivePipeline(
            pipeline_config,
            quantization=quantization,
            device=self.device,
            dtype=self.dtype,
        )

        if self.use_compile:
            self._apply_compile()

        print("Pipeline ready\n")

    def _apply_compile(self):
        """Apply torch.compile to VACE components."""
        print("Applying torch.compile to VACE components...")
        generator = self.pipeline.components.generator
        model = generator.model

        # 1. Compile VACE patch embedding
        if hasattr(model, "vace_patch_embedding"):
            model.vace_patch_embedding = torch.compile(
                model.vace_patch_embedding,
                mode="max-autotune-no-cudagraphs",
                dynamic=False,
            )
            print("  - Compiled vace_patch_embedding")

        # 2. Compile VACE blocks
        if hasattr(model, "vace_blocks"):
            for i, block in enumerate(model.vace_blocks):
                model.vace_blocks[i] = torch.compile(
                    block,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                )
            print(f"  - Compiled {len(model.vace_blocks)} vace_blocks")

        # 3. Compile VAE/TAE encoder and decoder
        # NOTE: TAE uses nn.Sequential with MemBlock layers that have special
        # iteration patterns and forward signatures (requires 'past' arg).
        # torch.compile breaks this - skip TAE compilation (already fast ~9-16ms).
        # For WanVAE, the Encoder3d/Decoder3d can be compiled directly.
        vae = self.pipeline.components.vae
        from scope.core.pipelines.wan2_1.vae.tae import TAEWrapper

        if isinstance(vae, TAEWrapper):
            print("  - Skipping TAE compile (MemBlock incompatible, already fast)")
        elif hasattr(vae, "model"):
            # WanVAE: compile Encoder3d and Decoder3d directly
            if hasattr(vae.model, "encoder"):
                vae.model.encoder = torch.compile(
                    vae.model.encoder,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                )
                print("  - Compiled WanVAE encoder")
            if hasattr(vae.model, "decoder"):
                vae.model.decoder = torch.compile(
                    vae.model.decoder,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                )
                print("  - Compiled WanVAE decoder")

        print("  torch.compile applied\n")

    def _create_depth_inputs(self, chunk_index):
        """Create synthetic depth map inputs for a chunk."""
        # Simulate depth maps: grayscale gradient replicated to 3 channels
        depth = (
            torch.rand(
                1,
                3,
                self.frames_per_chunk,
                self.height,
                self.width,
                device=self.device,
                dtype=self.dtype,
            )
            * 2.0
            - 1.0
        )  # [-1, 1] range
        return depth

    def _create_inpainting_inputs(self, chunk_index):
        """Create synthetic inpainting inputs for a chunk."""
        # Input video frames
        input_frames = (
            torch.rand(
                1,
                3,
                self.frames_per_chunk,
                self.height,
                self.width,
                device=self.device,
                dtype=self.dtype,
            )
            * 2.0
            - 1.0
        )

        # Circle mask in center (1=inpaint, 0=preserve)
        mask = torch.zeros(
            1,
            1,
            self.frames_per_chunk,
            self.height,
            self.width,
            device=self.device,
            dtype=self.dtype,
        )
        cy, cx = self.height // 2, self.width // 2
        radius = min(self.height, self.width) // 4
        y, x = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij",
        )
        circle = ((y - cy) ** 2 + (x - cx) ** 2) < radius**2
        mask[:, :, :, :, :] = circle.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return input_frames, mask

    def run_depth_benchmark(self, warmup=2, iterations=5):
        """Benchmark depth conditioning mode end-to-end."""
        print("=" * 60)
        print("  DEPTH CONDITIONING BENCHMARK")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Chunks: {self.num_chunks} x {self.frames_per_chunk} frames")
        print(f"  Warmup: {warmup}, Iterations: {iterations}")
        print("=" * 60)

        total_timing = self._get_timing("depth_total")
        chunk_timing = self._get_timing("depth_per_chunk")

        # Warmup
        print(f"\nWarmup ({warmup} runs)...")
        for w in range(warmup):
            self._run_depth_generation()
            print(f"  Warmup {w + 1}/{warmup} done")

        # Benchmark
        print(f"\nBenchmark ({iterations} runs)...")
        for i in range(iterations):
            chunk_times = self._run_depth_generation()
            total_time = sum(chunk_times)
            total_timing.times_ms.append(total_time * 1000)
            for ct in chunk_times:
                chunk_timing.times_ms.append(ct * 1000)

            total_frames = self.num_chunks * self.frames_per_chunk
            fps = total_frames / total_time
            print(
                f"  Run {i + 1}/{iterations}: {total_time:.2f}s, "
                f"{fps:.2f} FPS, chunks={[f'{t:.2f}s' for t in chunk_times]}"
            )

        self._print_depth_results()

    def _run_depth_generation(self):
        """Run one full depth generation, return per-chunk times."""
        chunk_times = []
        for chunk_idx in range(self.num_chunks):
            depth_input = self._create_depth_inputs(chunk_idx)
            kwargs = {
                "prompts": [{"text": "a cat walking", "weight": 100}],
                "vace_context_scale": 1.5,
                "vace_input_frames": depth_input,
            }
            start = time.perf_counter()
            self.pipeline(**kwargs)
            torch.cuda.synchronize()
            chunk_times.append(time.perf_counter() - start)
        return chunk_times

    def run_inpainting_benchmark(self, warmup=2, iterations=5):
        """Benchmark inpainting mode end-to-end."""
        print("\n" + "=" * 60)
        print("  INPAINTING BENCHMARK")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  Chunks: {self.num_chunks} x {self.frames_per_chunk} frames")
        print(f"  Warmup: {warmup}, Iterations: {iterations}")
        print("=" * 60)

        total_timing = self._get_timing("inpaint_total")
        chunk_timing = self._get_timing("inpaint_per_chunk")

        # Warmup
        print(f"\nWarmup ({warmup} runs)...")
        for w in range(warmup):
            self._run_inpainting_generation()
            print(f"  Warmup {w + 1}/{warmup} done")

        # Benchmark
        print(f"\nBenchmark ({iterations} runs)...")
        for i in range(iterations):
            chunk_times = self._run_inpainting_generation()
            total_time = sum(chunk_times)
            total_timing.times_ms.append(total_time * 1000)
            for ct in chunk_times:
                chunk_timing.times_ms.append(ct * 1000)

            total_frames = self.num_chunks * self.frames_per_chunk
            fps = total_frames / total_time
            print(
                f"  Run {i + 1}/{iterations}: {total_time:.2f}s, "
                f"{fps:.2f} FPS, chunks={[f'{t:.2f}s' for t in chunk_times]}"
            )

        self._print_inpainting_results()

    def _run_inpainting_generation(self):
        """Run one full inpainting generation, return per-chunk times."""
        chunk_times = []
        for chunk_idx in range(self.num_chunks):
            input_frames, mask = self._create_inpainting_inputs(chunk_idx)
            kwargs = {
                "prompts": [{"text": "a fireball", "weight": 100}],
                "vace_context_scale": 1.5,
                "vace_input_frames": input_frames,
                "vace_input_masks": mask,
            }
            start = time.perf_counter()
            self.pipeline(**kwargs)
            torch.cuda.synchronize()
            chunk_times.append(time.perf_counter() - start)
        return chunk_times

    def run_isolated_vace_benchmark(self, warmup=3, iterations=10):
        """Benchmark isolated VACE components (encoding, hint generation)."""
        print("\n" + "=" * 60)
        print("  ISOLATED VACE COMPONENT BENCHMARK")
        print("=" * 60)

        generator = self.pipeline.components.generator
        model = generator.model
        vae = self.pipeline.components.vae
        vae_dtype = next(vae.parameters()).dtype

        # ---- VAE Encode Benchmark ----
        print("\n--- VAE Encode (single stream, 12 frames) ---")
        frames = torch.rand(
            1,
            3,
            self.frames_per_chunk,
            self.height,
            self.width,
            device=self.device,
            dtype=vae_dtype,
        )
        vae_enc_timing = self._get_timing("isolated_vae_encode")

        for _ in range(warmup):
            vae.encode_to_latent(frames, use_cache=False)

        for _ in range(iterations):
            with cuda_timer(vae_enc_timing):
                vae.encode_to_latent(frames, use_cache=False)

        print(
            f"  Mean: {vae_enc_timing.mean_ms:.2f}ms, Std: {vae_enc_timing.std_ms:.2f}ms"
        )

        # ---- VAE Encode Dual Stream (inpainting) ----
        print("\n--- VAE Encode Dual Stream (inpainting, 2x12 frames) ---")
        inactive = torch.zeros_like(frames)
        reactive = frames.clone()
        dual_enc_timing = self._get_timing("isolated_vae_encode_dual")

        for _ in range(warmup):
            vae.encode_to_latent(inactive, use_cache=False)
            vae.encode_to_latent(reactive, use_cache=False)

        for _ in range(iterations):
            with cuda_timer(dual_enc_timing):
                vae.encode_to_latent(inactive, use_cache=False)
                vae.encode_to_latent(reactive, use_cache=False)

        print(
            f"  Mean: {dual_enc_timing.mean_ms:.2f}ms, Std: {dual_enc_timing.std_ms:.2f}ms"
        )
        overhead = dual_enc_timing.mean_ms - vae_enc_timing.mean_ms
        print(
            f"  Dual-stream overhead vs single: +{overhead:.2f}ms ({overhead / vae_enc_timing.mean_ms * 100:.1f}%)"
        )

        # ---- VACE Patch Embedding ----
        print("\n--- VACE Patch Embedding ---")
        # Create synthetic 96-channel VACE context
        latent_h = self.height // 8
        latent_w = self.width // 8
        latent_f = (self.frames_per_chunk + 3) // 4  # temporal downsample
        vace_ctx = torch.rand(
            96,
            latent_f,
            latent_h,
            latent_w,
            device=self.device,
            dtype=self.dtype,
        )
        patch_emb_timing = self._get_timing("isolated_vace_patch_emb")

        if hasattr(model, "vace_patch_embedding"):
            for _ in range(warmup):
                model.vace_patch_embedding(vace_ctx.unsqueeze(0))

            for _ in range(iterations):
                with cuda_timer(patch_emb_timing):
                    model.vace_patch_embedding(vace_ctx.unsqueeze(0))

            print(
                f"  Mean: {patch_emb_timing.mean_ms:.2f}ms, Std: {patch_emb_timing.std_ms:.2f}ms"
            )

        # ---- VACE Blocks Forward ----
        print("\n--- VACE Blocks Forward (hint generation) ---")
        if hasattr(model, "vace_blocks") and hasattr(model, "forward_vace"):
            # Build inputs for forward_vace
            # We need: x (main sequence), vace_context, seq_len, e, seq_lens, grid_sizes, freqs, context, context_lens, block_mask, crossattn_cache
            patch_size = model.patch_size
            seq_f = latent_f // patch_size[0]
            seq_h = latent_h // patch_size[1]
            seq_w = latent_w // patch_size[2]
            seq_len_val = seq_f * seq_h * seq_w

            x_dummy = torch.rand(
                1, seq_len_val, model.dim, device=self.device, dtype=self.dtype
            )
            e_dummy = torch.rand(
                1, seq_f, 6, model.dim, device=self.device, dtype=self.dtype
            )
            seq_lens = torch.tensor([seq_len_val], dtype=torch.long)
            grid_sizes = torch.tensor([[seq_f, seq_h, seq_w]], dtype=torch.long)
            context_dummy = torch.rand(
                1, 512, model.dim, device=self.device, dtype=self.dtype
            )
            crossattn_cache = [None] * model.num_layers

            vace_fwd_timing = self._get_timing("isolated_vace_forward")

            for _ in range(warmup):
                model.forward_vace(
                    x_dummy,
                    [vace_ctx],
                    seq_len_val,
                    e_dummy,
                    seq_lens,
                    grid_sizes,
                    model.causal_wan_model.freqs.to(self.device),
                    context_dummy,
                    None,
                    model.causal_wan_model.block_mask,
                    crossattn_cache,
                )

            for _ in range(iterations):
                with cuda_timer(vace_fwd_timing):
                    model.forward_vace(
                        x_dummy,
                        [vace_ctx],
                        seq_len_val,
                        e_dummy,
                        seq_lens,
                        grid_sizes,
                        model.causal_wan_model.freqs.to(self.device),
                        context_dummy,
                        None,
                        model.causal_wan_model.block_mask,
                        crossattn_cache,
                    )

            print(
                f"  Mean: {vace_fwd_timing.mean_ms:.2f}ms, Std: {vace_fwd_timing.std_ms:.2f}ms"
            )

        # ---- VAE Decode ----
        print("\n--- VAE Decode (3 latent frames) ---")
        # TAE decode expects [batch, frames, channels, height, width]
        latent_for_decode = torch.rand(
            1,
            latent_f,
            16,
            latent_h,
            latent_w,
            device=self.device,
            dtype=vae_dtype,
        )
        vae_dec_timing = self._get_timing("isolated_vae_decode")

        for _ in range(warmup):
            vae.decode_to_pixel(latent_for_decode, use_cache=False)

        for _ in range(iterations):
            with cuda_timer(vae_dec_timing):
                vae.decode_to_pixel(latent_for_decode, use_cache=False)

        print(
            f"  Mean: {vae_dec_timing.mean_ms:.2f}ms, Std: {vae_dec_timing.std_ms:.2f}ms"
        )

    def _print_depth_results(self):
        total = self.timings.get("depth_total")
        chunk = self.timings.get("depth_per_chunk")
        if not total:
            return
        print("\n--- Depth Conditioning Results ---")
        total_frames = self.num_chunks * self.frames_per_chunk
        fps = total_frames / (total.mean_ms / 1000)
        print(f"  Total: {total.mean_ms:.0f}ms ± {total.std_ms:.0f}ms")
        print(f"  Per-chunk: {chunk.mean_ms:.0f}ms ± {chunk.std_ms:.0f}ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Latency (first chunk): {chunk.min_ms:.0f}ms")

    def _print_inpainting_results(self):
        total = self.timings.get("inpaint_total")
        chunk = self.timings.get("inpaint_per_chunk")
        if not total:
            return
        print("\n--- Inpainting Results ---")
        total_frames = self.num_chunks * self.frames_per_chunk
        fps = total_frames / (total.mean_ms / 1000)
        print(f"  Total: {total.mean_ms:.0f}ms ± {total.std_ms:.0f}ms")
        print(f"  Per-chunk: {chunk.mean_ms:.0f}ms ± {chunk.std_ms:.0f}ms")
        print(f"  FPS: {fps:.2f}")
        print(f"  Latency (first chunk): {chunk.min_ms:.0f}ms")

    def print_summary(self, baseline_timings=None):
        """Print full benchmark summary, optionally comparing to baseline."""
        print("\n" + "=" * 80)
        print("  BENCHMARK SUMMARY")
        if self.use_compile:
            print("  Mode: torch.compile ENABLED")
        else:
            print("  Mode: BASELINE (no compile)")
        print("=" * 80)

        for name, timing in sorted(self.timings.items()):
            baseline = baseline_timings.get(name) if baseline_timings else None
            line = f"  {name:40s}: {timing.mean_ms:8.1f}ms ± {timing.std_ms:5.1f}ms"
            if baseline:
                speedup = baseline.mean_ms / timing.mean_ms if timing.mean_ms > 0 else 0
                pct = (
                    (1 - timing.mean_ms / baseline.mean_ms) * 100
                    if baseline.mean_ms > 0
                    else 0
                )
                line += f"  ({pct:+.1f}%, {speedup:.2f}x)"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="VACE Benchmark")
    parser.add_argument(
        "--mode", choices=["depth", "inpainting", "both", "isolated"], default="both"
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--fp8", action="store_true", help="Enable FP8 quantization")
    parser.add_argument(
        "--compare", action="store_true", help="Run both baseline and compile, compare"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=3)
    args = parser.parse_args()

    if args.compare:
        # Run baseline first, then compile, and compare
        print("\n" + "#" * 80)
        print("  PHASE 1: BASELINE (no torch.compile)")
        print("#" * 80)

        baseline = VACEBenchmark(
            use_compile=False,
            use_fp8=args.fp8,
            height=args.height,
            width=args.width,
            num_chunks=args.chunks,
        )
        _run_benchmarks(baseline, args)
        baseline_timings = baseline.timings

        # Clean up
        del baseline
        gc.collect()
        torch.cuda.empty_cache()

        print("\n" + "#" * 80)
        print("  PHASE 2: WITH torch.compile")
        print("#" * 80)

        compiled = VACEBenchmark(
            use_compile=True,
            use_fp8=args.fp8,
            height=args.height,
            width=args.width,
            num_chunks=args.chunks,
        )
        _run_benchmarks(compiled, args)

        compiled.print_summary(baseline_timings)
    else:
        bench = VACEBenchmark(
            use_compile=args.compile,
            use_fp8=args.fp8,
            height=args.height,
            width=args.width,
            num_chunks=args.chunks,
        )
        _run_benchmarks(bench, args)
        bench.print_summary()


def _run_benchmarks(bench, args):
    if args.mode in ("isolated", "both"):
        bench.run_isolated_vace_benchmark(
            warmup=args.warmup, iterations=args.iterations
        )
    if args.mode in ("depth", "both"):
        bench.run_depth_benchmark(warmup=args.warmup, iterations=args.iterations)
    if args.mode in ("inpainting", "both"):
        bench.run_inpainting_benchmark(warmup=args.warmup, iterations=args.iterations)


if __name__ == "__main__":
    main()
