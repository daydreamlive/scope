"""
Temporary test script to profile all LoRA merge modes.
Compares: permanent_merge, runtime_peft, gpu_reconstruct, and cuda_graph_recapture.
"""

import argparse
import time

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir
from lib.schema import Quantization
from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
from pipelines.memory import is_cuda_low_memory

LORA_PATH = "models/lora/wan14b/origami_000000500.safetensors"
ALL_MERGE_MODES = [
    "permanent_merge",
    "runtime_peft",
    "gpu_reconstruct",
    "cuda_graph_recapture",
]


def test_merge_mode(merge_mode: str):
    """Test a single LoRA merge mode and return profiling results."""
    print(f"\n{'='*80}")
    print(f"Testing merge mode: {merge_mode}")
    print(f"{'='*80}\n")

    config = OmegaConf.load("pipelines/krea_realtime_video/model.yaml")

    models_dir = get_models_dir()
    height = 228
    width = 228

    config["model_dir"] = str(models_dir)
    config["generator_path"] = str(
        get_model_file_path("krea-realtime-video/krea-realtime-video-14b.safetensors")
    )
    config["text_encoder_path"] = str(
        get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
    )
    config["tokenizer_path"] = str(
        get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
    )
    config["vae_path"] = str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    config["height"] = height
    config["width"] = width

    # Configure LoRA
    config["lora_merge_mode"] = merge_mode
    config["loras"] = [{"path": LORA_PATH, "scale": 1.0}]

    device = torch.device("cuda")

    # Profile: Pipeline initialization (includes LoRA loading)
    print("test_merge_mode: Starting pipeline initialization...")
    start_init = time.time()

    pipeline = KreaRealtimeVideoPipeline(
        config,
        low_memory=is_cuda_low_memory(device),
        quantization=Quantization.FP8_E4M3FN,  # Required for 14B model performance
        compile=False,  # No compilation for fair comparison of LoRA methods
        device=device,
        dtype=torch.bfloat16,
    )

    init_time = time.time() - start_init
    print(f"test_merge_mode: Pipeline init time: {init_time:.3f}s")

    # Profile: Prepare (first frame setup)
    prompt = "A person folding an origami crane in a Japanese garden, paper art style"
    print("test_merge_mode: Starting prepare...")
    start_prepare = time.time()

    pipeline.prepare(prompts=[{"text": prompt}], should_prepare=True)

    prepare_time = time.time() - start_prepare
    print(f"test_merge_mode: Prepare time: {prepare_time:.3f}s")

    # Collect video outputs
    video_outputs = []

    # Profile: First frame generation
    print("test_merge_mode: Generating first frame...")
    start_first_frame = time.time()

    output = pipeline()
    torch.cuda.synchronize()
    video_outputs.append(output.detach().cpu())

    first_frame_time = time.time() - start_first_frame
    num_frames = output.shape[0]
    print(
        f"test_merge_mode: First frame time: {first_frame_time:.3f}s ({num_frames} frames, {num_frames/first_frame_time:.2f} fps)"
    )

    # Profile: Subsequent frames (warmup + average)
    warmup_iterations = 3
    timing_iterations = 10

    print(f"test_merge_mode: Running {warmup_iterations} warmup iterations...")
    for i in range(warmup_iterations):
        _ = pipeline()
    torch.cuda.synchronize()

    print(f"test_merge_mode: Running {timing_iterations} timing iterations...")
    frame_times = []
    for i in range(timing_iterations):
        start = time.time()
        output = pipeline()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        frame_times.append(elapsed)
        num_frames = output.shape[0]
        video_outputs.append(output.detach().cpu())
        print(
            f"test_merge_mode:   Iteration {i+1}: {elapsed:.3f}s ({num_frames/elapsed:.2f} fps)"
        )

    avg_frame_time = sum(frame_times) / len(frame_times)
    avg_fps = num_frames / avg_frame_time
    print(
        f"test_merge_mode: Average frame time: {avg_frame_time:.3f}s ({avg_fps:.2f} fps)"
    )

    # Profile: LoRA scale update
    print("test_merge_mode: Testing LoRA scale update (1.0 -> 0.5)...")
    start_update = time.time()

    pipeline.prepare(
        lora_scales=[{"path": LORA_PATH, "scale": 0.5}], should_prepare=False
    )
    torch.cuda.synchronize()

    update_time = time.time() - start_update
    print(f"test_merge_mode: LoRA scale update time: {update_time:.3f}s")

    # Generate several frames after update to let KV cache catch up and show LoRA effects
    print(
        "test_merge_mode: Generating frames after scale update to let KV cache catch up..."
    )
    post_update_frames = 30
    start = time.time()
    for i in range(post_update_frames):
        output = pipeline()
        video_outputs.append(output.detach().cpu())
    torch.cuda.synchronize()
    post_update_time = time.time() - start
    avg_post_update = post_update_time / post_update_frames
    print(
        f"test_merge_mode: Post-update frames ({post_update_frames} frames): {post_update_time:.3f}s total, {avg_post_update:.3f}s avg ({num_frames/avg_post_update:.2f} fps)"
    )

    # Profile: Second LoRA scale update
    print("test_merge_mode: Testing second LoRA scale update (0.5 -> 0.8)...")
    start_update2 = time.time()

    pipeline.prepare(
        lora_scales=[{"path": LORA_PATH, "scale": 0.8}], should_prepare=False
    )
    torch.cuda.synchronize()

    update_time2 = time.time() - start_update2
    print(f"test_merge_mode: Second LoRA scale update time: {update_time2:.3f}s")

    # Generate several more frames after second update
    print("test_merge_mode: Generating frames after second scale update...")
    post_update2_frames = 30
    for i in range(post_update2_frames):
        output = pipeline()
        video_outputs.append(output.detach().cpu())
    torch.cuda.synchronize()
    print(
        f"test_merge_mode: Post-update2 frames: {post_update2_frames} frames generated"
    )

    # Save video output
    video_filename = f"pipelines/krea_realtime_video/output_{merge_mode}.mp4"
    print(f"test_merge_mode: Saving video to {video_filename}...")
    video_tensor = torch.cat(video_outputs, dim=0)
    video_np = video_tensor.contiguous().numpy()
    export_to_video(video_np, video_filename, fps=16)
    print(f"test_merge_mode: Video saved ({video_tensor.shape[0]} total frames)")

    # Cleanup
    del pipeline
    del video_outputs
    del video_tensor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Return results
    return {
        "merge_mode": merge_mode,
        "init_time": init_time,
        "prepare_time": prepare_time,
        "first_frame_time": first_frame_time,
        "avg_frame_time": avg_frame_time,
        "avg_fps": avg_fps,
        "update_time": update_time,
        "update_time2": update_time2,
        "post_update_time": post_update_time,
        "total_time": init_time + prepare_time + first_frame_time + sum(frame_times),
    }


def main():
    """Run tests for all merge modes and compare results."""
    parser = argparse.ArgumentParser(
        description="Profile LoRA merge modes for performance comparison"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "permanent_merge",
            "runtime_peft",
            "gpu_reconstruct",
            "cuda_graph_recapture",
        ],
        help="Which merge mode to test (default: all)",
    )
    args = parser.parse_args()

    # Determine which modes to test
    if args.mode == "all":
        merge_modes = ALL_MERGE_MODES
    else:
        merge_modes = [args.mode]

    print("=" * 80)
    print("LoRA Merge Mode Performance Comparison")
    print(f"LoRA: {LORA_PATH}")
    print(f"Testing: {', '.join(merge_modes)}")
    print("=" * 80)

    results = []
    for merge_mode in merge_modes:
        try:
            result = test_merge_mode(merge_mode)
            results.append(result)
        except Exception as e:
            print(f"\nmain: ERROR testing {merge_mode}: {e}")
            import traceback

            traceback.print_exc()

        # Short pause between tests
        time.sleep(2)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")

    if not results:
        print("main: No results to compare")
        return

    # Print header - dynamic column widths based on tested modes
    results_dict = {r["merge_mode"]: r for r in results}
    tested_modes = [mode for mode in ALL_MERGE_MODES if mode in results_dict]

    if len(tested_modes) == 1:
        # Single mode - simpler output
        mode = tested_modes[0]
        print(f"{'Metric':<30} | {'Value':<18}")
        print("-" * 52)
    else:
        # Multiple modes - comparison table
        header = f"{'Metric':<30} |"
        for mode in tested_modes:
            header += f" {mode:<18} |"
        print(header)
        print("-" * (32 + 21 * len(tested_modes)))

    # Print each metric
    metrics = [
        ("Init Time (s)", "init_time"),
        ("Prepare Time (s)", "prepare_time"),
        ("First Frame Time (s)", "first_frame_time"),
        ("Avg Frame Time (s)", "avg_frame_time"),
        ("Avg FPS", "avg_fps"),
        ("Update Time #1 (s)", "update_time"),
        ("Update Time #2 (s)", "update_time2"),
        ("Post-Update Frame (s)", "post_update_time"),
        ("Total Time (s)", "total_time"),
    ]

    for label, key in metrics:
        values = []
        for mode in tested_modes:
            if mode in results_dict:
                val = results_dict[mode][key]
                values.append(f"{val:.3f}")
            else:
                values.append("N/A")

        if len(tested_modes) == 1:
            print(f"{label:<30} | {values[0]:<18}")
        else:
            row = f"{label:<30} |"
            for val in values:
                row += f" {val:<18} |"
            print(row)

    # Calculate and print relative performance (only if multiple modes tested and permanent_merge is one of them)
    if len(tested_modes) > 1 and "permanent_merge" in results_dict:
        print("\n" + "=" * 80)
        print("RELATIVE PERFORMANCE (vs permanent_merge baseline)")
        print("=" * 80 + "\n")

        baseline = results_dict["permanent_merge"]
        comparison_modes = [m for m in tested_modes if m != "permanent_merge"]

        header = f"{'Metric':<30} |"
        for mode in comparison_modes:
            header += f" {mode:<18} |"
        print(header)
        print("-" * (32 + 21 * len(comparison_modes)))

        for label, key in metrics:
            if key == "avg_fps":
                # For FPS, higher is better
                baseline_val = baseline[key]
                values = []
                for mode in comparison_modes:
                    if mode in results_dict:
                        val = results_dict[mode][key]
                        relative = (val / baseline_val - 1) * 100
                        values.append(f"{relative:+.1f}%")
                    else:
                        values.append("N/A")
            else:
                # For time metrics, lower is better
                baseline_val = baseline[key]
                values = []
                for mode in comparison_modes:
                    if mode in results_dict:
                        val = results_dict[mode][key]
                        relative = (val / baseline_val - 1) * 100
                        values.append(f"{relative:+.1f}%")
                    else:
                        values.append("N/A")

            row = f"{label:<30} |"
            for val in values:
                row += f" {val:<18} |"
            print(row)

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if "permanent_merge" in results_dict and "runtime_peft" in results_dict:
        pm = results_dict["permanent_merge"]
        peft = results_dict["runtime_peft"]

        print("\nInference Performance (permanent_merge vs runtime_peft):")
        fps_overhead = (peft["avg_fps"] / pm["avg_fps"] - 1) * 100
        print(f"  - FPS overhead with PEFT: {fps_overhead:+.1f}%")

        print("\nUpdate Performance:")
        print("  - permanent_merge: NOT SUPPORTED (requires pipeline reload)")
        print(f"  - runtime_peft: {peft['update_time']:.3f}s (fast, <0.1s expected)")
        if "gpu_reconstruct" in results_dict:
            gr = results_dict["gpu_reconstruct"]
            print(
                f"  - gpu_reconstruct: {gr['update_time']:.3f}s (slower, ~2s expected)"
            )

        print("\nRecommendations:")
        print("  - For maximum FPS (fixed scale): permanent_merge")
        print("  - For fast updates with minimal overhead: runtime_peft")
        print("  - For zero overhead with occasional updates: gpu_reconstruct")
        if "cuda_graph_recapture" in results_dict:
            cg = results_dict["cuda_graph_recapture"]
            print(
                f"  - For graph replay with fast updates: cuda_graph_recapture ({cg['avg_fps']:.2f} fps, {cg['update_time']:.3f}s updates)"
            )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
