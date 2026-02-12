"""
Detailed profiling of inpainting pipeline to identify bottlenecks.

Usage:
    python -m scope.core.pipelines.longlive.profile_inpainting
"""

from functools import wraps
from pathlib import Path

import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline


class InstrumentedTimer:
    """Collects per-stage timings using CUDA events."""

    def __init__(self):
        self.records: dict[str, list[float]] = {}
        self._stack: list[tuple[str, torch.cuda.Event]] = []

    def start(self, name: str):
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self._stack.append((name, start_event))

    def stop(self, name: str):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == name:
                _, start_event = self._stack.pop(i)
                ms = start_event.elapsed_time(end_event)
                self.records.setdefault(name, []).append(ms)
                return
        raise ValueError(f"No start event for {name}")

    def report(self):
        print("\n" + "=" * 70)
        print("  DETAILED TIMING BREAKDOWN")
        print("=" * 70)

        # Calculate total from e2e if available
        total_ms = 0
        if "e2e_chunk" in self.records:
            total_ms = sum(self.records["e2e_chunk"]) / len(self.records["e2e_chunk"])

        for name, times in sorted(self.records.items()):
            mean = sum(times) / len(times)
            pct = (mean / total_ms * 100) if total_ms > 0 else 0
            print(f"  {name:40s}: {mean:8.2f}ms  ({pct:5.1f}%)")


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    height, width = 512, 512
    frames_per_chunk = 12

    print("=" * 70)
    print("  INPAINTING PROFILING")
    print("=" * 70)

    # Initialize pipeline
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

    print("Loading pipeline...")
    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=dtype)
    print("Pipeline ready\n")

    timer = InstrumentedTimer()

    # Instrument the key methods
    generator = pipeline.components.generator
    model = generator.model
    vae = pipeline.components.vae

    # Wrap generator.forward to measure total diffusion model time per step
    orig_gen_forward = generator.forward.__func__  # Get unbound method

    def timed_gen_forward(self_gen, *args, **kwargs):
        timer.start("generator_forward")
        result = orig_gen_forward(self_gen, *args, **kwargs)
        timer.stop("generator_forward")
        return result

    import types

    generator.forward = types.MethodType(timed_gen_forward, generator)

    # Wrap forward_vace to measure hint generation time
    orig_forward_vace = (
        model.forward_vace.__func__
        if hasattr(model.forward_vace, "__func__")
        else model.forward_vace
    )

    def timed_forward_vace(self_model, *args, **kwargs):
        timer.start("vace_forward_hints")
        result = orig_forward_vace(self_model, *args, **kwargs)
        timer.stop("vace_forward_hints")
        return result

    model.forward_vace = types.MethodType(timed_forward_vace, model)

    # Instrument pipeline blocks to find the 570ms overhead
    blocks = pipeline.blocks
    for block_name in [
        "text_conditioning",
        "embedding_blending",
        "set_timesteps",
        "setup_caches",
        "set_transformer_blocks_local_attn_size",
        "auto_preprocess_video",
        "auto_prepare_latents",
        "recache_frames",
        "vace_encoding",
        "denoise",
        "clean_kv_cache",
        "decode",
        "prepare_recache_frames",
        "prepare_next",
    ]:
        block = getattr(blocks, block_name, None)
        if block is not None and hasattr(block, "__call__"):
            orig_call = (
                block.__call__.__func__
                if hasattr(block.__call__, "__func__")
                else block.__call__
            )

            def make_timed_call(orig, name):
                def timed_call(self_block, *args, **kwargs):
                    timer.start(f"block_{name}")
                    result = orig(self_block, *args, **kwargs)
                    timer.stop(f"block_{name}")
                    return result

                return timed_call

            block.__call__ = types.MethodType(
                make_timed_call(orig_call, block_name), block
            )

    # Wrap VAE encode
    orig_encode = vae.encode_to_latent

    @wraps(orig_encode)
    def timed_encode(*args, **kwargs):
        timer.start("vae_encode")
        result = orig_encode(*args, **kwargs)
        timer.stop("vae_encode")
        return result

    vae.encode_to_latent = timed_encode

    # Wrap VAE decode
    orig_decode = vae.decode_to_pixel

    @wraps(orig_decode)
    def timed_decode(*args, **kwargs):
        timer.start("vae_decode")
        result = orig_decode(*args, **kwargs)
        timer.stop("vae_decode")
        return result

    vae.decode_to_pixel = timed_decode

    # Create synthetic inpainting inputs
    input_frames = (
        torch.rand(1, 3, frames_per_chunk, height, width, device=device, dtype=dtype)
        * 2.0
        - 1.0
    )
    mask = torch.zeros(
        1, 1, frames_per_chunk, height, width, device=device, dtype=dtype
    )
    cy, cx = height // 2, width // 2
    radius = min(height, width) // 4
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    circle = ((y - cy) ** 2 + (x - cx) ** 2) < radius**2
    mask[:, :, :, :, :] = circle.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)

    kwargs = {
        "prompts": [{"text": "a fireball", "weight": 100}],
        "vace_context_scale": 1.5,
        "vace_input_frames": input_frames,
        "vace_input_masks": mask,
    }

    # Warmup
    print("Warmup (2 runs)...")
    for _ in range(2):
        pipeline(**kwargs)
    timer.records.clear()

    # Profile
    num_runs = 5
    print(f"Profiling ({num_runs} runs)...")
    for i in range(num_runs):
        timer.start("e2e_chunk")
        pipeline(**kwargs)
        torch.cuda.synchronize()
        timer.stop("e2e_chunk")
        e2e = timer.records["e2e_chunk"][-1]
        print(f"  Run {i + 1}: {e2e:.1f}ms")

    timer.report()

    # Compute derived stats
    print("\n--- Derived Analysis ---")
    e2e = sum(timer.records["e2e_chunk"]) / len(timer.records["e2e_chunk"])
    vae_enc = sum(timer.records.get("vae_encode", [0])) / max(
        len(timer.records.get("vae_encode", [1])), 1
    )
    vae_dec = sum(timer.records.get("vae_decode", [0])) / max(
        len(timer.records.get("vae_decode", [1])), 1
    )
    vace_hints = sum(timer.records.get("vace_forward_hints", [0])) / max(
        len(timer.records.get("vace_forward_hints", [1])), 1
    )
    gen_fwd = sum(timer.records.get("generator_forward", [0])) / max(
        len(timer.records.get("generator_forward", [1])), 1
    )
    model_fwd = gen_fwd  # Alias for backward compat

    # Denoising loop calls model 4 times (4 timesteps by default)
    num_denoise_steps = 4
    denoise_per_step = model_fwd / num_denoise_steps if model_fwd > 0 else 0
    transformer_only = (
        model_fwd - vace_hints
    )  # hint gen happens once, transformer N times

    print(f"  E2E per chunk:          {e2e:.1f}ms")
    print(f"  VAE encode (all):       {vae_enc:.1f}ms ({vae_enc / e2e * 100:.1f}%)")
    print(f"  VAE decode:             {vae_dec:.1f}ms ({vae_dec / e2e * 100:.1f}%)")
    print(
        f"  VACE hint generation:   {vace_hints:.1f}ms ({vace_hints / e2e * 100:.1f}%)"
    )
    print(f"  Model forward (total):  {model_fwd:.1f}ms ({model_fwd / e2e * 100:.1f}%)")
    print(
        f"  Transformer blocks:     {transformer_only:.1f}ms ({transformer_only / e2e * 100:.1f}%)"
    )
    print(f"  Other overhead:         {e2e - vae_enc - vae_dec - model_fwd:.1f}ms")
    print(f"  Denoise steps:          {num_denoise_steps}")
    print(f"  Per denoise step:       ~{denoise_per_step:.1f}ms")

    # Compare depth vs inpainting
    print("\n--- Depth vs Inpainting Comparison ---")
    print("Running depth conditioning for comparison...")

    depth_timer = InstrumentedTimer()
    # Reuse same wrappers
    depth_input = (
        torch.rand(1, 3, frames_per_chunk, height, width, device=device, dtype=dtype)
        * 2.0
        - 1.0
    )
    depth_kwargs = {
        "prompts": [{"text": "a cat walking", "weight": 100}],
        "vace_context_scale": 1.5,
        "vace_input_frames": depth_input,
    }

    # Swap timer
    timer_backup = timer
    # Reset timer for depth runs
    timer.records.clear()

    for _ in range(2):
        pipeline(**depth_kwargs)
    timer.records.clear()

    for i in range(num_runs):
        timer.start("e2e_chunk")
        pipeline(**depth_kwargs)
        torch.cuda.synchronize()
        timer.stop("e2e_chunk")

    depth_e2e = sum(timer.records["e2e_chunk"]) / len(timer.records["e2e_chunk"])
    depth_vae_enc = sum(timer.records.get("vae_encode", [0])) / max(
        len(timer.records.get("vae_encode", [1])), 1
    )

    print(f"  Depth E2E per chunk:      {depth_e2e:.1f}ms")
    print(f"  Depth VAE encode:         {depth_vae_enc:.1f}ms")
    print(f"  Inpainting E2E per chunk: {e2e:.1f}ms")
    print(f"  Inpainting VAE encode:    {vae_enc:.1f}ms")
    print(
        f"  Inpainting overhead:      {e2e - depth_e2e:.1f}ms ({(e2e - depth_e2e) / depth_e2e * 100:.1f}%)"
    )
    print(f"  VAE encode overhead:      {vae_enc - depth_vae_enc:.1f}ms (dual-stream)")


if __name__ == "__main__":
    main()
