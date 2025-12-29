"""
Extension mode test script with scaling VACE context scale.

This script tests extension mode with a VACE context scale that scales from
minimum to maximum across N chunks:
- First chunk: Uses 'firstframe' mode with first_frame_image
- Subsequent chunks (1 to N-1): Uses 'lastframe' mode with last_frame_image,
  with VACE context scale scaling from min to max

Usage:
    Edit the CONFIG dictionary below to set paths and parameters.
    python -m scope.core.pipelines.longlive.test_vace_extension_scale
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== INPUT PATHS =====
    "first_frame_image": "frontend/public/assets/example.png",  # First frame reference
    "last_frame_image": "frontend/public/assets/example1.png",  # Last frame reference
    # ===== GENERATION PARAMETERS =====
    "prompt": "",  # Text prompt (can be empty for extension mode)
    "num_chunks": 8,  # Number of generation chunks
    "frames_per_chunk": 12,  # Frames per chunk (12 = 3 latent * 4 temporal upsample)
    "height": 512,
    "width": 512,
    # ===== VACE CONTEXT SCALE PARAMETERS =====
    "vace_context_scale_min": 0.0,  # Minimum VACE context scale (first chunk)
    "vace_context_scale_max": 0.5,  # Maximum VACE context scale (last chunk)
    "interpolation_mode": "weak_middle",  # Interpolation mode: "linear", "ease_in", "ease_out", "ease_in_out", "exponential", "logarithmic", "cosine", "strong_middle", "weak_middle"
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


def add_vace_scale_overlay(
    frames: np.ndarray,
    vace_scales_per_chunk: list[float],
    frames_per_chunk: int,
) -> np.ndarray:
    """
    Add VACE context scale overlay to video frames.

    Args:
        frames: Video frames [F, H, W, C] in [0, 1]
        vace_scales_per_chunk: List of VACE scales, one per chunk
        frames_per_chunk: Number of frames per chunk

    Returns:
        Frames with overlay [F, H, W, C] in [0, 1]
    """
    num_frames, height, width, channels = frames.shape
    overlayed_frames = []

    # Try to load a font, fall back to default if not available
    try:
        # Try to use a larger font if available
        font = ImageFont.truetype("arial.ttf", size=24)
    except OSError:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=24
            )
        except OSError:
            # Fall back to default font
            font = ImageFont.load_default()

    for frame_idx in range(num_frames):
        # Determine which chunk this frame belongs to
        chunk_idx = frame_idx // frames_per_chunk
        if chunk_idx >= len(vace_scales_per_chunk):
            chunk_idx = len(vace_scales_per_chunk) - 1

        vace_scale = vace_scales_per_chunk[chunk_idx]

        # Convert frame to PIL Image
        frame_uint8 = (frames[frame_idx] * 255).astype(np.uint8)
        pil_image = Image.fromarray(frame_uint8)

        # Create draw context
        draw = ImageDraw.Draw(pil_image)

        # Prepare text
        text = f"VACE Scale: {vace_scale:.3f}"
        chunk_text = f"Chunk: {chunk_idx}"

        # Get text bounding boxes
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        chunk_bbox = draw.textbbox((0, 0), chunk_text, font=font)
        chunk_text_width = chunk_bbox[2] - chunk_bbox[0]
        chunk_text_height = chunk_bbox[3] - chunk_bbox[1]

        # Position text at top-left with padding
        padding = 10
        text_x = padding
        text_y = padding

        # Draw semi-transparent background rectangles
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Background for scale text
        overlay_draw.rectangle(
            [
                text_x - 5,
                text_y - 5,
                text_x + text_width + 5,
                text_y + text_height + 5,
            ],
            fill=(0, 0, 0, 180),  # Semi-transparent black
        )

        # Background for chunk text
        overlay_draw.rectangle(
            [
                text_x - 5,
                text_y + text_height + 5,
                text_x + chunk_text_width + 5,
                text_y + text_height + chunk_text_height + 10,
            ],
            fill=(0, 0, 0, 180),  # Semi-transparent black
        )

        # Composite overlay onto image
        pil_image = Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert(
            "RGB"
        )

        # Redraw text on the composited image
        draw = ImageDraw.Draw(pil_image)
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
        draw.text(
            (text_x, text_y + text_height + 5),
            chunk_text,
            fill=(255, 255, 255),
            font=font,
        )

        # Convert back to numpy
        frame_overlayed = np.array(pil_image).astype(np.float32) / 255.0
        overlayed_frames.append(frame_overlayed)

    return np.array(overlayed_frames)


def apply_interpolation(t: float, mode: str) -> float:
    """
    Apply interpolation function to normalized time t [0, 1].

    Args:
        t: Normalized time (0 to 1)
        mode: Interpolation mode string

    Returns:
        Interpolated value [0, 1]
    """
    if mode == "linear":
        return t
    elif mode == "ease_in":
        # Quadratic ease-in: slow start, fast end
        return t * t
    elif mode == "ease_out":
        # Quadratic ease-out: fast start, slow end
        return 1 - (1 - t) * (1 - t)
    elif mode == "ease_in_out":
        # Quadratic ease-in-out: slow start and end, fast middle
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - 2 * (1 - t) * (1 - t)
    elif mode == "exponential":
        # Exponential: very slow start, very fast end
        if t == 0:
            return 0
        if t == 1:
            return 1
        # Normalized exponential: (2^(10*t) - 1) / (2^10 - 1)
        return (2 ** (10 * t) - 1) / (2**10 - 1)
    elif mode == "logarithmic":
        # Logarithmic: fast start, very slow end
        if t == 0:
            return 0
        if t == 1:
            return 1
        # Normalized logarithmic: log(10*t + 1) / log(11)
        return np.log(10 * t + 1) / np.log(11)
    elif mode == "cosine":
        # Cosine: smooth S-curve
        return 1 - np.cos(t * np.pi / 2)
    elif mode == "strong_middle":
        # Strong middle: emphasizes middle values (higher in middle)
        # Uses a curve that peaks in the middle while still reaching 1 at the end
        if t == 0:
            return 0
        if t == 1:
            return 1
        # Use sin(πt/2) as base, then add a bell curve centered at 0.5
        # The bell curve is sin(πt) which peaks at 0.5
        base = np.sin(t * np.pi / 2)
        bell = np.sin(t * np.pi)
        # Blend: more bell in middle, ensuring we don't exceed bounds
        # At t=0.5: base ≈ 0.707, bell = 1, so blend ≈ 0.707 + 0.2*1 = 0.907
        # At t=1: base = 1, bell = 0, so blend = 1
        return base + 0.2 * bell * (1 - base)
    elif mode == "weak_middle":
        # Weak middle: de-emphasizes middle values (lower in middle)
        # Uses a curve that dips in the middle while still reaching 1 at the end
        if t == 0:
            return 0
        if t == 1:
            return 1
        # Linear base with middle dip: t * (1 - 0.5*sin(πt))
        # sin(πt) is 0 at t=0,1 and peaks at t=0.5, so this creates a dip
        return t * (1 - 0.5 * np.sin(t * np.pi))
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")


def calculate_vace_scale(
    chunk_index: int,
    num_chunks: int,
    scale_min: float,
    scale_max: float,
    interpolation_mode: str = "linear",
) -> float:
    """
    Calculate VACE context scale for a given chunk.

    Scales from scale_min to scale_max using the specified interpolation mode.

    Args:
        chunk_index: Current chunk index (0-based)
        num_chunks: Total number of chunks
        scale_min: Minimum scale value (first chunk)
        scale_max: Maximum scale value (last chunk)
        interpolation_mode: Interpolation mode ("linear", "ease_in", "ease_out", etc.)

    Returns:
        VACE context scale for this chunk
    """
    if num_chunks == 1:
        return scale_max

    # Normalized time from 0 to 1
    t = chunk_index / (num_chunks - 1)

    # Apply interpolation function
    t_interpolated = apply_interpolation(t, interpolation_mode)

    # Scale from min to max
    scale = scale_min + t_interpolated * (scale_max - scale_min)
    return scale


# ============================= MAIN =============================


def main():
    print("=" * 80)
    print("  LongLive Extension Mode - Scaling VACE Context Scale Test")
    print("=" * 80)

    # Parse configuration
    config = CONFIG

    print("\nConfiguration:")
    print(
        f"  Extension mode: firstframe (chunk 0), then lastframe (chunks 1-{config['num_chunks'] - 1})"
    )
    print(f"  Prompt: '{config['prompt']}'")
    print(f"  Chunks: {config['num_chunks']} x {config['frames_per_chunk']} frames")
    print(f"  Resolution: {config['height']}x{config['width']}")
    print(
        f"  VACE Scale: {config['vace_context_scale_min']} -> {config['vace_context_scale_max']}"
    )
    print(f"  Interpolation: {config['interpolation_mode']}")

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
            "height": config["height"],
            "width": config["width"],
        }
    )

    # Set vace_in_dim for extension mode (masked encoding: 32 + 64 = 96 channels)
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
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

    # Generate video
    print("=== Generating Video ===")
    outputs = []
    latency_measures = []
    fps_measures = []
    vace_scales_used = []

    frames_per_chunk = config["frames_per_chunk"]
    num_chunks = config["num_chunks"]

    for chunk_index in range(num_chunks):
        start_time = time.time()

        # Determine if this is first chunk
        is_first_chunk = chunk_index == 0

        # Prepare pipeline kwargs
        kwargs = {
            "prompts": [{"text": config["prompt"], "weight": 100}],
        }

        # First chunk: use first_frame_image with scale of 1.0
        if is_first_chunk:
            kwargs["extension_mode"] = "firstframe"
            kwargs["first_frame_image"] = first_frame_image
            kwargs["vace_context_scale"] = 1.0
            vace_scales_used.append(1.0)
            extension_info = "first"
        else:
            # Subsequent chunks: use last_frame_image with scaling VACE context scale
            # Scale from min to max across chunks 1 to N-1
            # For chunk i (where i >= 1), scale from 0 to 1 across remaining chunks
            num_subsequent_chunks = num_chunks - 1
            chunk_position = (
                chunk_index - 1
            )  # Position within subsequent chunks (0 to num_subsequent_chunks - 1)

            vace_scale = calculate_vace_scale(
                chunk_position,
                num_subsequent_chunks,
                config["vace_context_scale_min"],
                config["vace_context_scale_max"],
                config["interpolation_mode"],
            )
            vace_scales_used.append(vace_scale)

            kwargs["extension_mode"] = "lastframe"
            kwargs["last_frame_image"] = last_frame_image
            kwargs["vace_context_scale"] = vace_scale
            extension_info = f"last (scale={vace_scale:.3f})"

        print(
            f"Chunk {chunk_index}/{num_chunks - 1}: "
            f"VACE scale={kwargs['vace_context_scale']:.3f}, "
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
    # output_video is already [F, H, W, C] from pipeline
    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    # Add VACE scale overlay
    print("Adding VACE scale overlay to frames...")
    output_video_np = add_vace_scale_overlay(
        output_video_np,
        vace_scales_used,
        frames_per_chunk,
    )

    output_filename = (
        f"output_extension_scale_"
        f"{config['vace_context_scale_min']:.2f}to{config['vace_context_scale_max']:.2f}_"
        f"{config['interpolation_mode']}_"
        f"{num_chunks}chunks.mp4"
    )
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

    print("\n=== VACE Context Scale Progression ===")
    for chunk_idx, scale in enumerate(vace_scales_used):
        print(f"  Chunk {chunk_idx}: {scale:.4f}")

    print("\n" + "=" * 80)
    print("  Test Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Main output: {output_filename}")
    print(
        f"VACE context scale scaled from {config['vace_context_scale_min']:.2f} "
        f"to {config['vace_context_scale_max']:.2f} across {num_chunks} chunks"
    )


if __name__ == "__main__":
    main()
