import argparse
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.pipelines.utils import parse_jsonl_prompts, print_statistics

from .pipeline import MemFlowPipeline


def generate_video(
    pipeline: MemFlowPipeline,
    prompt_texts: list[str],
    output_path: Path,
    max_output_frames: int = 81,
) -> tuple[list[float], list[float]]:
    """Generate a video from a sequence of prompts.

    Args:
        pipeline: The MemFlowPipeline instance
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
    parser = argparse.ArgumentParser(description="Test MemFlow pipeline")
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
            "generator_path": str(get_model_file_path("MemFlow/base.pt")),
            "lora_path": str(get_model_file_path("MemFlow/lora.pt")),
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
    pipeline = MemFlowPipeline(config, device=device, dtype=torch.bfloat16)

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
            "A realistic video of a Texas Hold'em poker event at a casino. A male player in his late 30s with a medium build, short dark hair, light stubble, and a sharp jawline wears a fitted navy blazer over a charcoal crew-neck tee, dark jeans, and a stainless-steel watch. He sits at a well-lit poker table and tightly grips his hole cards, wearing a tense, serious expression. The table is filled with chips of various colors, the dealer is seen dealing cards, and several rows of slot machines glow in the background. The camera focuses on the player's strained concentration. Wide shot to medium close-up.",
            "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble, sharp jawline—dressed in a fitted navy blazer over a charcoal tee, dark jeans, and a stainless-steel watch—flicks his cards onto the felt, then leans back in the chair with arms spread wide in celebration. The dealer continues dealing to the table as stacks of multicolored chips crowd the surface; slot machines and nearby patrons fill the background. The camera locks onto the player's exuberant reaction. Wide shot to medium close-up.",
            "A realistic video of a Texas Hold'em poker event at a casino. The same late-30s male player, medium build with short dark hair and light stubble, wearing a navy blazer, charcoal tee, dark jeans, and a stainless-steel watch, reveals the winning hand and leans back in celebration while the dealer keeps the game moving. A nearby patron claps and cheers for the winner, amplifying the festive atmosphere. The table brims with colorful chips, with slot machines and other tables behind. The camera centers on the winner's reaction as the applause rises. Wide shot to medium close-up.",
            "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble—still in his navy blazer, charcoal tee, dark jeans, and stainless-steel watch—sits upright and begins neatly arranging the stacks of chips in front of him, methodically straightening and organizing the piles. The dealer continues dealing, and rows of slot machines pulse in the background. The camera captures the composed, purposeful movements at the well-lit table. Wide shot to medium close-up.",
            "A realistic video of a Texas Hold'em poker event at a casino. The same late-30s male player with short dark hair, light stubble, and a sharp jawline, wearing a fitted navy blazer over a charcoal tee, dark jeans, and a stainless-steel watch, glances over his chips and breaks into a proud, self-assured smile, basking in the victorious moment. Multicolored chips crowd the felt, the dealer works the table, and slot machines glow behind. The camera emphasizes the winner's pride and satisfaction. Wide shot to medium close-up.",
            "A realistic video of a Texas Hold'em poker event at a casino. The same male player—late 30s, medium build, short dark hair, light stubble—dressed in a navy blazer, charcoal tee, dark jeans, and a stainless-steel watch—shares a celebratory high-five with a nearby patron after the win, laughter and cheers rippling around the table. Stacks of chips are spread across the felt, the dealer continues dealing, and the background features rows of slot machines and other patrons. The camera focuses on the jubilant interaction. Wide shot to medium close-up.",
        ]

        output_path = output_dir / "output.mp4"
        latency_measures, fps_measures = generate_video(
            pipeline, prompt_texts, output_path
        )
        print(f"Saved video to {output_path}")
        print_statistics(latency_measures, fps_measures)


if __name__ == "__main__":
    main()
