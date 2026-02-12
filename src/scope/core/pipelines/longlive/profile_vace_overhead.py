"""
Profile VACE forward to measure kernel launch overhead vs actual GPU compute.

Compares:
1. Normal forward_vace execution (includes all Python/dispatch overhead)
2. torch.profiler breakdown of CPU vs CUDA time for forward_vace specifically

This helps estimate the theoretical benefit of CUDA graphs.

Usage:
    python -m scope.core.pipelines.longlive.profile_vace_overhead
"""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    height, width = 512, 512
    frames_per_chunk = 12

    print("Loading pipeline...")
    script_dir = Path(__file__).parent
    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "vace_path": str(
                get_model_file_path(
                    "Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"
                )
            ),
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
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=dtype)
    print("Pipeline ready\n")

    # Get the model for direct forward_vace calls
    model = pipeline.components.generator.model

    # We need to set up the inputs that forward_vace expects
    # First, run a full pipeline call to get the model into a valid state
    input_frames = (
        torch.rand(1, 3, frames_per_chunk, height, width, device=device, dtype=dtype)
        * 2.0
        - 1.0
    )
    kwargs = {
        "prompts": [{"text": "a fireball", "weight": 100}],
        "vace_context_scale": 1.5,
        "vace_input_frames": input_frames,
    }

    # Warmup pipeline to populate caches and compile flex_attention
    print("Warmup pipeline (2 runs)...")
    for _ in range(2):
        pipeline(**kwargs)

    # Capture forward_vace arguments by patching the actual method on the class

    captured_args = {}
    # The model may be wrapped by PeftModel -> CausalVaceWanModel
    # Walk through wrappers to find the class that has forward_vace
    target_model = model
    while not hasattr(type(target_model), "forward_vace"):
        if hasattr(target_model, "base_model"):
            target_model = target_model.base_model
        elif hasattr(target_model, "model"):
            target_model = target_model.model
        elif hasattr(target_model, "causal_wan_model"):
            target_model = target_model.causal_wan_model
        else:
            break
    model_class = type(target_model)
    print(f"Patching forward_vace on {model_class.__name__}")
    orig_forward_vace = model_class.forward_vace

    def capturing_forward_vace(self_model, *args, **kw):
        if not captured_args:
            captured_args["args"] = args
            captured_args["kwargs"] = kw
        return orig_forward_vace(self_model, *args, **kw)

    model_class.forward_vace = capturing_forward_vace

    # Run once to capture args
    pipeline(**kwargs)

    # Restore original
    model_class.forward_vace = orig_forward_vace

    if not captured_args:
        print("ERROR: Failed to capture forward_vace arguments")
        return

    fv_args = captured_args["args"]
    fv_kwargs = captured_args["kwargs"]

    print("Captured forward_vace inputs:")
    for i, a in enumerate(fv_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: {a.shape} {a.dtype}")
        elif isinstance(a, list):
            print(f"  arg[{i}]: list of {len(a)} items")
        else:
            print(f"  arg[{i}]: {type(a).__name__} = {a}")

    # --- Benchmark 1: Wall-clock timing of forward_vace ---
    print("\n=== Wall-clock timing (CUDA events) ===")
    # Warmup
    for _ in range(5):
        model.forward_vace(*fv_args, **fv_kwargs)
    torch.cuda.synchronize()

    num_runs = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(num_runs):
        start_event.record()
        model.forward_vace(*fv_args, **fv_kwargs)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    mean_ms = sum(times) / len(times)
    print(f"  forward_vace: {mean_ms:.2f}ms (mean of {num_runs} runs)")
    print(f"  Per denoising step contribution: {mean_ms:.2f}ms")
    print(f"  Per chunk (4 steps): {mean_ms * 4:.2f}ms")

    # --- Benchmark 2: torch.profiler for CPU vs CUDA breakdown ---
    print("\n=== torch.profiler breakdown (forward_vace only) ===")

    # Warmup
    for _ in range(3):
        model.forward_vace(*fv_args, **fv_kwargs)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
    ) as prof:
        for _ in range(5):
            model.forward_vace(*fv_args, **fv_kwargs)
        torch.cuda.synchronize()

    # Aggregate stats
    events = prof.key_averages()

    total_cpu_time = 0
    total_cuda_time = 0
    total_calls = 0

    print("\n  Top 15 operations by CUDA time:")
    table = events.table(sort_by="cuda_time_total", row_limit=15)
    print(table)

    print("\n  Top 10 operations by CPU time:")
    table = events.table(sort_by="cpu_time_total", row_limit=10)
    print(table)

    # Count kernel launches
    for evt in events:
        total_cpu_time += evt.cpu_time_total
        total_cuda_time += getattr(
            evt, "cuda_time_total", getattr(evt, "self_cuda_time_total", 0)
        )
        total_calls += evt.count

    print("\n  Summary (5 forward_vace calls):")
    print(f"    Total CPU time: {total_cpu_time / 1000:.1f}ms")
    print(f"    Total CUDA time: {total_cuda_time / 1000:.1f}ms")
    print(f"    Total op calls: {total_calls}")
    print(f"    Per-call CPU time: {total_cpu_time / 1000 / 5:.1f}ms")
    print(f"    Per-call CUDA time: {total_cuda_time / 1000 / 5:.1f}ms")

    # --- Benchmark 3: Count actual kernel launches ---
    print("\n=== Kernel launch count ===")

    kernel_count = 0
    for evt in events:
        if evt.device_type is not None and "cuda" in str(evt.device_type).lower():
            kernel_count += evt.count

    # More precise: count events that have CUDA time
    cuda_events = [
        e
        for e in events
        if getattr(e, "cuda_time_total", getattr(e, "self_cuda_time_total", 0)) > 0
    ]
    print(f"  Unique CUDA operations: {len(cuda_events)}")
    print(
        f"  Total CUDA kernel invocations (across 5 calls): {sum(e.count for e in cuda_events)}"
    )
    print(
        f"  Per forward_vace call: ~{sum(e.count for e in cuda_events) // 5} kernel launches"
    )

    # --- Estimate CUDA graph benefit ---
    print("\n=== CUDA Graph Benefit Estimate ===")
    wall_clock_ms = mean_ms
    cuda_compute_ms = total_cuda_time / 1000 / 5  # per call
    kernels_per_call = sum(e.count for e in cuda_events) // 5

    # Kernel launch overhead estimate: ~5-10μs per launch on Windows
    launch_overhead_ms = kernels_per_call * 0.007  # 7μs average from profiling

    print(f"  Wall-clock per call: {wall_clock_ms:.2f}ms")
    print(f"  CUDA compute per call: {cuda_compute_ms:.2f}ms")
    print(f"  Estimated launch overhead: {launch_overhead_ms:.2f}ms")
    print(f"  Overhead ratio: {launch_overhead_ms / wall_clock_ms * 100:.1f}%")
    print("")
    print(f"  CUDA graphs would save ~{launch_overhead_ms:.1f}ms per forward_vace call")
    print(f"  Per chunk (4 denoising steps): ~{launch_overhead_ms * 4:.1f}ms")
    print(f"  As % of chunk time (~735ms): ~{launch_overhead_ms * 4 / 735 * 100:.1f}%")


if __name__ == "__main__":
    main()
