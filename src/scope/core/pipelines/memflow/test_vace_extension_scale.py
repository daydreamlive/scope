"""
Extension mode test script for MemFlow pipeline.

This script runs three separate tests:
1. Firstframe mode only - all chunks use first_frame_image
2. Lastframe mode only - all chunks use last_frame_image
3. Firstframe + lastframe - first chunk uses first_frame_image, subsequent chunks use last_frame_image

Each test produces its own video output.

Usage:
    Edit the CONFIG dictionary below to set paths and parameters.
    python -m scope.core.pipelines.memflow.test_vace_extension_scale
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import MemFlowPipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== INPUT PATHS =====
    "first_frame_image": "frontend/public/assets/example.png",  # First frame reference
    "last_frame_image": "frontend/public/assets/example1.png",  # Last frame reference
    # ===== GENERATION PARAMETERS =====
    "prompt": "",  # Text prompt (can be empty for extension mode)
    "num_chunks": 8,  # Number of generation chunks
    "frames_per_chunk": 12,  # Frames per chunk (12 = 3 latent * 4 temporal upsample)
    "height": 480,
    "width": 832,
    "vace_context_scale": 1.0,  # VACE context scale for all chunks
    # ===== OUTPUT =====
    "output_dir": "vace_tests/extension_scale",  # path/to/output_dir
}

# ========================= END CONFIGURATION =========================

# ============================= UTILITIES =============================


def resolve_path(path_str: str, relative_to: Path) -> Path:
    """Resolve path relative to a base directory or as absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def generate_test_video(
    pipeline: MemFlowPipeline,
    test_name: str,
    config: dict,
    first_frame_image: str,
    last_frame_image: str,
    output_dir: Path,
) -> tuple[str, list[float], list[float]]:
    """
    Generate a video for a specific test mode.

    Args:
        pipeline: The MemFlowPipeline instance
        test_name: Name of the test ("firstframe", "lastframe", "firstframelastframe")
        config: Configuration dictionary
        first_frame_image: Path to first frame image
        last_frame_image: Path to last frame image
        output_dir: Output directory for videos

    Returns:
        Tuple of (output_filename, latency_measures, fps_measures)
    """
    print(f"\n{'=' * 80}")
    print(f"  Test: {test_name}")
    print(f"{'=' * 80}")

    outputs = []
    latency_measures = []
    fps_measures = []
    frames_per_chunk = config["frames_per_chunk"]
    num_chunks = config["num_chunks"]

    for chunk_index in range(num_chunks):
        start_time = time.time()

        # Prepare pipeline kwargs
        kwargs = {
            "prompts": [{"text": config["prompt"], "weight": 100}],
            "vace_context_scale": config["vace_context_scale"],
        }

        # Determine extension mode based on test name
        if test_name == "firstframe":
            # All chunks use firstframe mode
            kwargs["extension_mode"] = "firstframe"
            kwargs["first_frame_image"] = first_frame_image
            extension_info = "firstframe"
        elif test_name == "lastframe":
            # All chunks use lastframe mode
            kwargs["extension_mode"] = "lastframe"
            kwargs["last_frame_image"] = last_frame_image
            extension_info = "lastframe"
        elif test_name == "firstframelastframe":
            # First chunk uses firstframe, subsequent chunks use lastframe
            if chunk_index == 0:
                kwargs["extension_mode"] = "firstframe"
                kwargs["first_frame_image"] = first_frame_image
                extension_info = "firstframe"
            else:
                kwargs["extension_mode"] = "lastframe"
                kwargs["last_frame_image"] = last_frame_image
                extension_info = "lastframe"
        else:
            raise ValueError(f"Unknown test name: {test_name}")

        print(
            f"Chunk {chunk_index}/{num_chunks - 1}: "
            f"VACE scale={config['vace_context_scale']:.3f}, "
            f"frames={frames_per_chunk}, "
            f"extension={extension_info}"
        )

        # Generate
        output = pipeline(**kwargs)

        # Metrics
        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start_time
        fps = num_output_frames / latency

        print(
            f"  Generated {num_output_frames} frames, "
            f"latency={latency:.2f}s, fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs.append(output.detach().cpu())

    # Concatenate outputs
    output_video = torch.concat(outputs)

    print(f"\nFinal output shape: {output_video.shape}")

    # Convert to numpy and clip
    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    # Save video
    output_filename = f"output_{test_name}_{num_chunks}chunks.mp4"
    output_path = output_dir / output_filename
    export_to_video(output_video_np, output_path, fps=16)

    print(f"\nSaved output: {output_path}")

    # Statistics
    print("\n=== Performance Statistics ===")
    print(
        f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Max: {max(latency_measures):.2f}s, "
        f"Min: {min(latency_measures):.2f}s"
    )
    print(
        f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, "
        f"Min: {min(fps_measures):.2f}"
    )

    return output_filename, latency_measures, fps_measures


# ============================= MAIN =============================


def main():
    print("=" * 80)
    print("  MemFlow Extension Mode Test Suite")
    print("=" * 80)

    # Parse configuration
    config = CONFIG

    print("\nConfiguration:")
    print(f"  Prompt: '{config['prompt']}'")
    print(f"  Chunks: {config['num_chunks']} x {config['frames_per_chunk']} frames")
    print(f"  Resolution: {config['height']}x{config['width']}")
    print(f"  VACE Context Scale: {config['vace_context_scale']}")

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent
    output_dir = resolve_path(config["output_dir"], script_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"  Output: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # Initialize pipeline
    print("Initializing pipeline...")

    vace_path = str(
        get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
    )

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(get_model_file_path("MemFlow/base.pt")),
            "lora_path": str(get_model_file_path("MemFlow/lora.pt")),
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
            "height": config["height"],
            "width": config["width"],
        }
    )

    # Set vace_in_dim for extension mode (masked encoding: 32 + 64 = 96 channels)
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = MemFlowPipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline ready\n")

    # Load frame images for Extension mode
    print("=== Preparing Extension Inputs ===")

    # Load first_frame_image
    first_frame_path = resolve_path(config["first_frame_image"], project_root)
    if not first_frame_path.exists():
        raise FileNotFoundError(f"First frame image not found: {first_frame_path}")
    first_frame_image = str(first_frame_path)
    print(f"  First frame image: {first_frame_path}")

    # Load last_frame_image
    last_frame_path = resolve_path(config["last_frame_image"], project_root)
    if not last_frame_path.exists():
        raise FileNotFoundError(f"Last frame image not found: {last_frame_path}")
    last_frame_image = str(last_frame_path)
    print(f"  Last frame image: {last_frame_path}")
    print()

    # Run three tests
    test_results = []

    # Test 1: Firstframe only
    output_filename, latency_measures, fps_measures = generate_test_video(
        pipeline,
        "firstframe",
        config,
        first_frame_image,
        last_frame_image,
        output_dir,
    )
    test_results.append(
        {
            "name": "firstframe",
            "filename": output_filename,
            "latency": latency_measures,
            "fps": fps_measures,
        }
    )

    # Test 2: Lastframe only
    output_filename, latency_measures, fps_measures = generate_test_video(
        pipeline,
        "lastframe",
        config,
        first_frame_image,
        last_frame_image,
        output_dir,
    )
    test_results.append(
        {
            "name": "lastframe",
            "filename": output_filename,
            "latency": latency_measures,
            "fps": fps_measures,
        }
    )

    # Test 3: Firstframe + Lastframe
    output_filename, latency_measures, fps_measures = generate_test_video(
        pipeline,
        "firstframelastframe",
        config,
        first_frame_image,
        last_frame_image,
        output_dir,
    )
    test_results.append(
        {
            "name": "firstframelastframe",
            "filename": output_filename,
            "latency": latency_measures,
            "fps": fps_measures,
        }
    )

    # Summary
    print("\n" + "=" * 80)
    print("  Test Suite Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\n=== Summary ===")
    for result in test_results:
        avg_latency = sum(result["latency"]) / len(result["latency"])
        avg_fps = sum(result["fps"]) / len(result["fps"])
        print(
            f"  {result['name']}: {result['filename']} "
            f"(Avg Latency: {avg_latency:.2f}s, Avg FPS: {avg_fps:.2f})"
        )


if __name__ == "__main__":
    main()
