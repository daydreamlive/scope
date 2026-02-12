"""
Torch profiler for inpainting pipeline.

Usage:
    python -m scope.core.pipelines.longlive.profile_torch
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

    # Create inpainting inputs
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
    print("Warmup...")
    for _ in range(2):
        pipeline(**kwargs)

    # Profile
    print("Profiling with torch.profiler...")
    output_dir = Path("vace_tests/profiles")
    output_dir.mkdir(exist_ok=True, parents=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        pipeline(**kwargs)
        torch.cuda.synchronize()

    # Print top CUDA operations
    print("\n=== Top 30 CUDA Operations by Total Time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n=== Top 20 CPU Operations by Total Time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Export chrome trace
    trace_path = output_dir / "inpainting_trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace exported to: {trace_path}")
    print("Open in chrome://tracing or https://ui.perfetto.dev/")


if __name__ == "__main__":
    main()
