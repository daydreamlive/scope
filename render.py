
# individual timeline
# uv run render.py timelines/0.01bias.json -o timelines/output --width 416 --height 240
# all timelines in directory
# uv run render.py timelines/ -o timelines/output --width 416 --height 240


import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_unique_output_path(output_path: str) -> str:
    """Get a unique output path by appending a number if file exists."""
    path = Path(output_path)

    if not path.exists():
        return output_path

    # Split into stem and suffix
    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    # Find next available number
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return str(new_path)
        counter += 1


def discover_timeline_files(path: str) -> list[Path]:
    """Discover timeline JSON files from a path (file or directory)."""
    path_obj = Path(path)

    if path_obj.is_file():
        if path_obj.suffix.lower() != ".json":
            raise ValueError(f"Timeline file must be a JSON file: {path}")
        return [path_obj]

    if path_obj.is_dir():
        json_files = sorted(path_obj.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in directory: {path}")
        logger.info(f"Found {len(json_files)} timeline files in {path}")
        return json_files

    raise ValueError(f"Path does not exist: {path}")


def load_timeline(timeline_path: Path) -> dict:
    """Load and validate timeline JSON file."""
    logger.debug(f"Loading timeline from {timeline_path}")
    with open(timeline_path) as f:
        timeline = json.load(f)

    if "prompts" not in timeline:
        raise ValueError(f"Timeline must contain 'prompts' field: {timeline_path}")
    if "settings" not in timeline:
        raise ValueError(f"Timeline must contain 'settings' field: {timeline_path}")

    return timeline


def group_timelines_by_pipeline(
    timeline_files: list[Path],
) -> dict[str, list[tuple[Path, dict]]]:
    """Load and group timeline files by pipeline ID."""
    grouped = defaultdict(list)

    for timeline_file in timeline_files:
        try:
            timeline = load_timeline(timeline_file)
            pipeline_id = timeline["settings"]["pipelineId"]
            grouped[pipeline_id].append((timeline_file, timeline))
        except Exception as e:
            logger.error(f"Failed to load timeline {timeline_file}: {e}")
            continue

    return dict(grouped)


def _setup_base_config(config, settings, default_height: int, default_width: int, override_width: int | None = None, override_height: int | None = None):
    """Apply common config settings: model_dir, resolution, and seed."""
    models_dir = get_models_dir()
    config["model_dir"] = str(models_dir)

    resolution = settings.get("resolution", {})

    # Use override values if provided, otherwise use JSON values or defaults
    if override_height is not None:
        config["height"] = override_height
    else:
        config["height"] = resolution.get("height", default_height)

    if override_width is not None:
        config["width"] = override_width
    else:
        config["width"] = resolution.get("width", default_width)

    config["seed"] = settings.get("seed", 42)


def load_pipeline(pipeline_id: str, settings: dict, override_width: int | None = None, override_height: int | None = None):
    """Load the specified pipeline with settings."""
    logger.info(f"Loading pipeline: {pipeline_id}")

    if override_width is not None or override_height is not None:
        logger.info(f"Using resolution overrides: width={override_width}, height={override_height}")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    if pipeline_id == "longlive":
        from pipelines.longlive.pipeline import LongLivePipeline
        from pipelines.memory import is_cuda_low_memory

        config = OmegaConf.load("pipelines/longlive/model.yaml")
        _setup_base_config(config, settings, default_height=320, default_width=576, override_width=override_width, override_height=override_height)

        config["generator_path"] = get_model_file_path(
            "LongLive-1.3B/models/longlive_base.pt"
        )
        config["lora_path"] = get_model_file_path("LongLive-1.3B/models/lora.pt")
        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )

        pipeline = LongLivePipeline(
            config,
            low_memory=is_cuda_low_memory(device),
            device=device,
            dtype=dtype,
        )

    elif pipeline_id == "krea-realtime-video":
        from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

        config = OmegaConf.load("pipelines/krea_realtime_video/model.yaml")
        _setup_base_config(config, settings, default_height=512, default_width=512, override_width=override_width, override_height=override_height)

        config["generator_path"] = str(
            get_model_file_path(
                "krea-realtime-video/krea-realtime-video-14b.safetensors"
            )
        )
        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )
        config["tokenizer_path"] = str(
            get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
        )
        config["vae_path"] = str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))

        quantization = settings.get("quantization", None)

        # Force bias to 1.0 during initialization to avoid flex_attention compilation issues during warmup
        config["kv_cache_attention_bias"] = 1.0

        pipeline = KreaRealtimeVideoPipeline(
            config,
            quantization=quantization,
            compile=any(
                x in torch.cuda.get_device_name(0).lower() for x in ("h100", "hopper")
            ),
            device=device,
            dtype=dtype,
        )

    elif pipeline_id == "streamdiffusionv2":
        from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        config = OmegaConf.load("pipelines/streamdiffusionv2/model.yaml")
        _setup_base_config(config, settings, default_height=512, default_width=512, override_width=override_width, override_height=override_height)

        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )

        pipeline = StreamDiffusionV2Pipeline(config, device=device, dtype=dtype)

    else:
        raise ValueError(f"Unsupported pipeline: {pipeline_id}")

    logger.info(f"Pipeline {pipeline_id} loaded successfully")
    return pipeline


def render_timeline_with_pipeline(
    pipeline,
    timeline: dict,
    timeline_file: Path,
    output_path: str,
    fps: int = 16,
    kv_cache_attention_bias: float | None = None,
    manage_cache: bool | None = None,
    denoising_step_list: list[int] | None = None,
    noise_scale: float | None = None,
    noise_controller: bool | None = None,
):
    """Render a single timeline using a pre-loaded pipeline.

    Cache is automatically reset at the start of each timeline (first prompt block).
    """
    prompts = timeline["prompts"]

    # Process each prompt block
    outputs = []
    total_duration = prompts[-1]["endTime"] if prompts else 0
    total_frames_needed = int(total_duration * fps)
    logger.info(
        f"Rendering {timeline_file.name}: {len(prompts)} prompt blocks, {total_duration:.2f}s ({total_frames_needed} frames at {fps} fps)"
    )

    for i, prompt_block in enumerate(prompts):
        start_time = prompt_block["startTime"]
        end_time = prompt_block["endTime"]
        duration = end_time - start_time

        if duration <= 0:
            logger.warning(
                f"  Skipping prompt block {i} with zero or negative duration"
            )
            continue

        # Get prompts from block
        block_prompts = prompt_block.get("prompts", [])
        if not block_prompts:
            logger.warning(f"  Skipping prompt block {i} with no prompts")
            continue

        # Calculate number of frames needed for this duration
        frames_needed = int(duration * fps)

        logger.info(
            f"  Block {i+1}/{len(prompts)}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s, {frames_needed} frames)"
        )
        logger.debug(f"    Prompt: {block_prompts[0]['text'][:80]}...")

        # Prepare pipeline for this prompt - EXACTLY like the UI does
        # Reset cache on first block of each timeline (like pressing rewind in UI)
        should_prepare = i == 0
        is_first_block = i == 0

        # Build prepare kwargs with ALL parameters the UI passes
        prepare_kwargs = {"should_prepare": should_prepare}

        # Handle transitions vs direct prompt updates
        transition_steps = prompt_block.get("transitionSteps", 0)
        temporal_interpolation_method = prompt_block.get(
            "temporalInterpolationMethod", "linear"
        )

        # CRITICAL: First block cannot use transitions (no previous prompt to transition from)
        # Subsequent blocks can use transitions if transitionSteps > 0
        if not is_first_block and transition_steps > 0:
            # Use transition for smooth blending between prompts
            prepare_kwargs["transition"] = {
                "target_prompts": block_prompts,
                "num_steps": transition_steps,
                "temporal_interpolation_method": temporal_interpolation_method,
            }
        else:
            # First block or no transition - set prompts directly
            prepare_kwargs["prompts"] = block_prompts
            if temporal_interpolation_method:
                prepare_kwargs["prompt_interpolation_method"] = (
                    temporal_interpolation_method
                )

        # Krea-specific parameters - pass EVERYTHING
        if kv_cache_attention_bias is not None:
            prepare_kwargs["kv_cache_attention_bias"] = kv_cache_attention_bias

        if manage_cache is not None:
            prepare_kwargs["manage_cache"] = manage_cache

        if denoising_step_list is not None:
            prepare_kwargs["denoising_step_list"] = denoising_step_list

        if noise_scale is not None:
            prepare_kwargs["noise_scale"] = noise_scale

        if noise_controller is not None:
            prepare_kwargs["noise_controller"] = noise_controller

        pipeline.prepare(**prepare_kwargs)

        # Generate frames for this duration
        num_frames = 0

        block_start = time.time()
        while num_frames < frames_needed:
            output = pipeline()

            num_output_frames, _, _, _ = output.shape
            num_frames += num_output_frames
            outputs.append(output.detach().cpu())

        block_duration = time.time() - block_start
        generation_fps = num_frames / block_duration if block_duration > 0 else 0
        logger.info(
            f"    Generated {num_frames} frames in {block_duration:.2f}s ({generation_fps:.2f} fps)"
        )

    # Concatenate all outputs
    logger.info(f"  Concatenating {len(outputs)} chunks...")
    output_video = torch.concat(outputs)
    logger.info(
        f"  Total frames: {output_video.shape[0]} (expected {total_frames_needed})"
    )

    # Export to video
    logger.info(f"  Exporting to {output_path}...")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_video_np = output_video.contiguous().numpy()
    export_to_video(output_video_np, output_path, fps=fps)

    logger.info(f"  Video saved to {output_path}")


def determine_output_path(
    timeline_file: Path, output_arg: str | None, is_single_file: bool
) -> str:
    """Determine the output path for a timeline file."""
    # Default: use timeline filename in same directory
    if output_arg is None:
        return str(timeline_file.parent / f"{timeline_file.stem}.mp4")

    output_path = Path(output_arg)

    # If output is a directory, use timeline filename in that directory
    if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
        return str(output_path / f"{timeline_file.stem}.mp4")

    # If processing single file and output is a file path, use it directly
    if is_single_file:
        return str(output_path)

    # If processing multiple files, treat as directory
    return str(output_path / f"{timeline_file.stem}.mp4")


def render_timelines(timeline_path: str, output_arg: str | None = None, fps: int = 16, override_width: int | None = None, override_height: int | None = None):
    """Render timeline(s) from a file or directory, grouped by pipeline for efficiency."""
    # Discover timeline files
    timeline_files = discover_timeline_files(timeline_path)
    is_single_file = len(timeline_files) == 1

    # Group by pipeline
    grouped_timelines = group_timelines_by_pipeline(timeline_files)
    logger.info(
        f"Grouped {len(timeline_files)} timeline(s) into {len(grouped_timelines)} pipeline(s)"
    )

    # Create output directory if specified and it doesn't exist
    if output_arg:
        output_path = Path(output_arg)
        if output_path.is_dir() or (not output_path.suffix and not is_single_file):
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_path}")

    # Process each pipeline group
    for pipeline_id, timeline_list in sorted(grouped_timelines.items()):
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Processing {len(timeline_list)} timeline(s) for pipeline: {pipeline_id}"
        )
        logger.info(f"{'='*60}")

        # Load pipeline once for all timelines
        first_timeline = timeline_list[0][1]
        pipeline = load_pipeline(pipeline_id, first_timeline["settings"], override_width=override_width, override_height=override_height)

        # Render each timeline with the loaded pipeline
        for idx, (timeline_file, timeline) in enumerate(timeline_list):
            # Extract ALL relevant settings from timeline - match UI behavior exactly
            settings = timeline["settings"]

            # Initialize all possible parameters
            kv_cache_attention_bias = None
            manage_cache = None
            denoising_step_list = None
            noise_scale = None
            noise_controller = None

            if pipeline_id == "krea-realtime-video":
                # Get kv_cache_attention_bias from timeline settings
                # New scale: 0.01 to 1.0, where 1.0 = no bias (disabled)
                # Handle both old (negativeAttentionBias) and new (kvCacheAttentionBias) field names
                kv_cache_attention_bias = settings.get("kvCacheAttentionBias") or settings.get("negativeAttentionBias")

                manage_cache = settings.get("manageCache")
                denoising_step_list = settings.get("denoisingSteps")
                noise_scale = settings.get("noiseScale")
                noise_controller = settings.get("noiseController")

            # Determine output path
            output_path = determine_output_path(
                timeline_file, output_arg, is_single_file
            )

            # Get unique path to avoid overwriting
            output_path = get_unique_output_path(output_path)

            # Render
            try:
                render_timeline_with_pipeline(
                    pipeline,
                    timeline,
                    timeline_file,
                    output_path,
                    fps=fps,
                    kv_cache_attention_bias=kv_cache_attention_bias,
                    manage_cache=manage_cache,
                    denoising_step_list=denoising_step_list,
                    noise_scale=noise_scale,
                    noise_controller=noise_controller,
                )
            except Exception as e:
                logger.error(f"Failed to render {timeline_file}: {e}")
                import traceback

                traceback.print_exc()
                continue

        logger.info(f"Completed all timelines for pipeline: {pipeline_id}\n")


def main():
    """Main entry point for timeline rendering."""
    parser = argparse.ArgumentParser(
        description="Render timeline JSON file(s) to video (offline mode). "
        "Accepts a single file or directory. When processing a directory, "
        "timelines are grouped by pipeline for efficiency."
    )
    parser.add_argument(
        "timeline_path",
        help="Path to timeline JSON file or directory containing timeline files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file or directory path. "
        "For single file: defaults to {timeline-name}.mp4 in same directory. "
        "For directory: defaults to same directory as timelines. "
        "Can specify a directory or (for single file) a specific filename.",
    )
    parser.add_argument(
        "--fps", type=int, default=16, help="Output video framerate (default: 16)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override video width (ignores JSON settings)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override video height (ignores JSON settings)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error(
            "CUDA is not available. This script requires a CUDA-compatible GPU."
        )
        sys.exit(1)

    # Render timeline(s)
    try:
        render_timelines(args.timeline_path, args.output, args.fps, args.width, args.height)
        logger.info("\nAll timelines rendered successfully!")
    except Exception as e:
        logger.error(f"Failed to render timelines: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
