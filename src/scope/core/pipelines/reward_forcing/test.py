import argparse
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.utils import parse_jsonl_prompts, print_statistics

from .pipeline import RewardForcingPipeline


def generate_video(
    pipeline: RewardForcingPipeline,
    prompt_texts: list[str],
    output_path: Path,
    max_output_frames: int = 81,
) -> tuple[list[float], list[float]]:
    """Generate a video from a sequence of prompts.

    Args:
        pipeline: The RewardForcingPipeline instance
        prompt_texts: List of prompt strings
        output_path: Path to save the output video
        max_output_frames: Maximum frames to generate per prompt

    Returns:
        Tuple of (latency_measures, fps_measures)
    """
    outputs = []
    latency_measures = []
    fps_measures = []
    is_first_call = True

    for prompt_text in prompt_texts:
        num_frames = 0
        while num_frames < max_output_frames:
            start = time.time()

            prompts = [{"text": prompt_text, "weight": 100}]
            # Reset cache on first call of each video generation
            output = pipeline(prompts=prompts, init_cache=is_first_call)
            is_first_call = False

            num_output_frames, _, _, _ = output.shape
            latency = time.time() - start
            fps = num_output_frames / latency

            print(
                f"Pipeline generated {num_output_frames} frames latency={latency:2f}s fps={fps}"
            )

            latency_measures.append(latency)
            fps_measures.append(fps)
            num_frames += num_output_frames
            outputs.append(output.detach().cpu())

    # Concatenate all of the THWC tensors
    output_video = torch.concat(outputs)
    print(output_video.shape)
    output_video_np = output_video.contiguous().numpy()
    export_to_video(output_video_np, output_path, fps=16)

    return latency_measures, fps_measures


def main():
    parser = argparse.ArgumentParser(description="Test RewardForcing pipeline")
    parser.add_argument(
        "--prompts",
        type=str,
        help="Path to a JSONL file containing prompt sequences",
    )
    args = parser.parse_args()

    # Setup config and pipeline
    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("Reward-Forcing-T2V-1.3B/rewardforcing.pt")
            ),
            "text_encoder_path": str(
                get_model_file_path(
                    "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                )
            ),
            "tokenizer_path": str(
                get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
            ),
            "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
            "height": 480,
            "width": 832,
        }
    )

    device = torch.device("cuda")
    pipeline = RewardForcingPipeline(config, device=device, dtype=torch.bfloat16)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.prompts:
        # Load prompts from JSONL file
        prompt_sequences = parse_jsonl_prompts(args.prompts)
        print(f"Loaded {len(prompt_sequences)} prompt sequences from {args.prompts}")

        all_latency_measures = []
        all_fps_measures = []

        for i, prompt_texts in enumerate(prompt_sequences):
            print(f"\n=== Generating video {i} ({len(prompt_texts)} prompts) ===")
            output_path = output_dir / f"output_{i}.mp4"

            latency_measures, fps_measures = generate_video(
                pipeline, prompt_texts, output_path
            )
            all_latency_measures.extend(latency_measures)
            all_fps_measures.extend(fps_measures)

            print(f"Saved video to {output_path}")

        print_statistics(all_latency_measures, all_fps_measures)
    else:
        # Use default prompts
        prompt_texts = [
            "A bird's-eye shot of a glass bowl filled with clear water, perfectly still and transparent.",
            "In the same overhead view, a drop of black ink falls into the water, blooming into ethereal wisps and tendrils.",
        ]

        output_path = output_dir / "output.mp4"
        latency_measures, fps_measures = generate_video(
            pipeline, prompt_texts, output_path
        )
        print(f"Saved video to {output_path}")
        print_statistics(latency_measures, fps_measures)


if __name__ == "__main__":
    main()
