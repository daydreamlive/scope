"""
Point-Based Subject Control Test Script for LongLive Pipeline with VACE.

This implements "Point-Based Subject Control" as shown in VACE community demos.

## How It Works

Two-phase approach using extension mode + layout control:

**Chunk 0: Extension Mode (firstframe)**
- Establishes subject identity via first_frame_image
- Subject is encoded and cached in KV for subsequent chunks
- No trajectory control on this chunk

**Chunks 1+: Layout Control**
- vace_input_frames = Layout signal (white bg + black contour at trajectory position)
- vace_input_masks = Spatial masks indicating WHERE to generate
- Subject identity comes from KV cache, NOT from vace_input_frames

## Key Implementation Details

VaceEncodingBlock Priority (from vace_encoding.py):
1. extension mode (first_frame_image) - HIGHEST PRIORITY
2. conditioning mode (vace_input_frames)
3. R2V mode (vace_ref_images only)

Extension mode and conditioning mode are MUTUALLY EXCLUSIVE in the same chunk.
Do NOT provide first_frame_image and vace_input_frames together!

## Layout Control Format (from LayoutMaskAnnotator)

- White background (255, 255, 255)
- Black contour showing where subject should be positioned
- This is structural guidance, not the subject itself

Sources:
- VACE Project: https://ali-vilab.github.io/VACE-Page/
- MickMumpitz Guide: https://mickmumpitz.ai/guides/create-controllable-characters-for-your-ai-movies
- ComfyUI VACE: https://docs.comfy.org/tutorials/video/wan/vace
- VACE Paper: https://arxiv.org/abs/2503.07598

Usage:
    python -m scope.core.pipelines.longlive.test_point_based_subject_control
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
from PIL import Image

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline


def add_fps_overlay(
    frames: np.ndarray,
    fps: float,
    chunk_idx: int,
    total_chunks: int,
) -> np.ndarray:
    """
    Add FPS overlay to frames with semi-transparent background.

    Format: "XX.X FPS | Chunk N/M"
    """
    frames = frames.copy()
    text = f"{fps:.1f} FPS | Chunk {chunk_idx + 1}/{total_chunks}"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    thickness = 1
    color = (255, 255, 255)

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    padding = 8
    x, y = padding, text_h + padding + 2

    for i in range(len(frames)):
        frame = frames[i]
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (padding - 4, padding - 2),
            (x + text_w + 4, y + baseline + 4),
            (0, 0, 0),
            -1,
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.putText(
            frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )
        frames[i] = frame

    return frames


# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== SUBJECT INPUT =====
    # The subject image - establishes identity in first frame
    "subject_image": "frontend/public/assets/woman1.jpg",
    # ===== TRAJECTORY PARAMETERS =====
    # Pattern: "nod", "shake", "bounce", "circular", "figure8", "pendulum"
    "trajectory_pattern": "nod",
    # Trajectory center and amplitude (normalized 0-1 coordinates)
    "trajectory": {
        "center_x": 0.5,  # Center of trajectory
        "center_y": 0.35,  # Typically head region
        "amplitude_x": 0.08,  # Horizontal movement range
        "amplitude_y": 0.06,  # Vertical movement range
        "frequency": 2.0,  # Cycles per video
    },
    # ===== LAYOUT CONTROL PARAMETERS =====
    # Size of the trajectory contour in layout control frames
    "layout_radius": 80,  # Contour radius in pixels
    # ===== GENERATION PARAMETERS =====
    "prompt": "",
    "num_chunks": 15,
    "frames_per_chunk": 12,
    "height": 512,
    "width": 512,
    "vace_context_scale": 1.5,
    # ===== OUTPUT =====
    "output_dir": "vace_tests/point_based_control",
    "vae_type": "tae",
    "save_intermediates": True,
}

# ========================= END CONFIGURATION =========================


def create_trajectory_positions(
    num_frames: int,
    pattern: str,
    center_x: float,
    center_y: float,
    amplitude_x: float,
    amplitude_y: float,
    frequency: float,
) -> list[tuple[float, float]]:
    """
    Generate normalized trajectory positions (0-1 range) for each frame.

    Returns list of (x, y) tuples in normalized coordinates.
    """
    positions = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)  # Normalized time [0, 1]

        if pattern == "nod":
            # Head nodding - gentle up/down with slight sway
            x = center_x + amplitude_x * 0.3 * math.sin(4 * math.pi * frequency * t)
            y = center_y + amplitude_y * math.sin(2 * math.pi * frequency * t)

        elif pattern == "shake":
            # Head shaking - side to side
            x = center_x + amplitude_x * math.sin(2 * math.pi * frequency * t)
            y = center_y + amplitude_y * 0.2 * math.sin(4 * math.pi * frequency * t)

        elif pattern == "bounce":
            # Bouncing motion
            x = center_x
            y = center_y + amplitude_y * abs(math.sin(2 * math.pi * frequency * t))

        elif pattern == "circular":
            # Circular motion
            x = center_x + amplitude_x * math.cos(2 * math.pi * frequency * t)
            y = center_y + amplitude_y * math.sin(2 * math.pi * frequency * t)

        elif pattern == "figure8":
            # Figure-8 pattern
            x = center_x + amplitude_x * math.sin(2 * math.pi * frequency * t)
            y = center_y + amplitude_y * math.sin(4 * math.pi * frequency * t) * 0.5

        elif pattern == "pendulum":
            # Damped pendulum swing
            damping = 1 - t * 0.5
            x = center_x + amplitude_x * damping * math.sin(2 * math.pi * frequency * t)
            y = center_y + amplitude_y * 0.2 * abs(
                math.cos(2 * math.pi * frequency * t)
            )

        else:
            # Default simple oscillation
            x = center_x + amplitude_x * math.sin(2 * math.pi * frequency * t)
            y = center_y

        positions.append((x, y))

    return positions


def create_trajectory_visualization(
    positions: list[tuple[float, float]],
    width: int,
    height: int,
    ball_radius: int = 20,
) -> np.ndarray:
    """
    Create visual representation of trajectory (bouncing ball video).
    This is for visualization only - not used as VACE input.

    Returns [F, H, W, 3] uint8 array.
    """
    frames = []
    for x_norm, y_norm in positions:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        px = int(x_norm * width)
        py = int(y_norm * height)
        px = max(ball_radius, min(width - ball_radius, px))
        py = max(ball_radius, min(height - ball_radius, py))
        cv2.circle(frame, (px, py), ball_radius, (255, 255, 255), -1)
        frames.append(frame)
    return np.array(frames)


def create_layout_masks(
    num_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Create masks for layout control (V2V).

    For layout control (like depth, pose, layout), masks should be ALL ONES.
    The layout signal guides WHERE to place content, but the entire frame
    is generated (not partial inpainting).

    Returns [F, H, W] float32 array of ones.
    """
    # Layout control = generate everything, guided by layout
    # This is V2V control, not MV2V inpainting
    return np.ones((num_frames, height, width), dtype=np.float32)


def create_layout_control_frames(
    positions: list[tuple[float, float]],
    width: int,
    height: int,
    mask_radius: int,
) -> np.ndarray:
    """
    Create LAYOUT CONTROL frames for trajectory guidance.

    Layout control format (from LayoutMaskAnnotator):
    - White background (255, 255, 255)
    - Black contour showing object position

    This is a CONTROL SIGNAL, not inpainting - it tells the model
    WHERE to place the subject, not WHAT the subject looks like.

    Returns [F, H, W, C] uint8 array.
    """
    num_frames = len(positions)
    frames = np.ones((num_frames, height, width, 3), dtype=np.uint8) * 255  # White bg

    for i, (x_norm, y_norm) in enumerate(positions):
        px = int(x_norm * width)
        py = int(y_norm * height)
        px = max(mask_radius, min(width - mask_radius, px))
        py = max(mask_radius, min(height - mask_radius, py))

        # Draw black contour (not filled) - matches LayoutMaskAnnotator
        cv2.circle(frames[i], (px, py), mask_radius, (0, 0, 0), thickness=3)

    return frames


def preprocess_frames_for_vace(
    frames: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess frames for VACE input.

    Converts [F, H, W, C] uint8 [0, 255] to [1, C, F, H, W] float [-1, 1].
    """
    # Normalize to [0, 1] then to [-1, 1]
    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor * 2.0 - 1.0  # [-1, 1]

    # Rearrange: [F, H, W, C] -> [1, C, F, H, W]
    tensor = tensor.permute(0, 3, 1, 2)  # [F, C, H, W]
    tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]

    return tensor.to(device)


def preprocess_masks_for_vace(
    masks: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess masks for VACE input.

    Converts [F, H, W] float [0, 1] to [1, 1, F, H, W] float.
    """
    tensor = torch.from_numpy(masks).float()  # [F, H, W]
    tensor = tensor.unsqueeze(1)  # [F, 1, H, W]
    tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 1, F, H, W]
    return tensor.to(device)


def load_and_resize_image(
    image_path: str,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Load image and resize to target dimensions."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_width, target_height), Image.LANCZOS)
    return np.array(img)


def resolve_path(path_str: str, relative_to: Path) -> Path:
    """Resolve path relative to base directory."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def main():
    print("=" * 80)
    print("  Point-Based Subject Control Test")
    print("  (Trajectory -> Animated Masks -> VACE Inpainting)")
    print("=" * 80)

    config = CONFIG
    pattern = config["trajectory_pattern"]

    print(f"\nTrajectory pattern: {pattern}")
    print(f"Subject image: {config['subject_image']}")

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent
    output_dir = resolve_path(config["output_dir"], script_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Output directory: {output_dir}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load subject image
    print("Loading subject image...")
    subject_path = resolve_path(config["subject_image"], project_root)
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject image not found: {subject_path}")

    subject_image = load_and_resize_image(
        str(subject_path),
        config["height"],
        config["width"],
    )
    print(f"  Subject shape: {subject_image.shape}")

    # Calculate total frames
    total_frames = config["num_chunks"] * config["frames_per_chunk"]
    print(f"  Total frames: {total_frames}")

    # Generate trajectory positions
    print(f"\nGenerating '{pattern}' trajectory...")
    traj_config = config["trajectory"]
    positions = create_trajectory_positions(
        num_frames=total_frames,
        pattern=pattern,
        center_x=traj_config["center_x"],
        center_y=traj_config["center_y"],
        amplitude_x=traj_config["amplitude_x"],
        amplitude_y=traj_config["amplitude_y"],
        frequency=traj_config["frequency"],
    )
    print(f"  Generated {len(positions)} trajectory points")
    print(f"  First: ({positions[0][0]:.3f}, {positions[0][1]:.3f})")
    print(f"  Last: ({positions[-1][0]:.3f}, {positions[-1][1]:.3f})")

    # Create masks for layout control (all ones = generate everything)
    print("\nCreating layout masks...")
    masks = create_layout_masks(
        num_frames=total_frames,
        height=config["height"],
        width=config["width"],
    )
    print("  Masks: all ones (V2V layout control, not MV2V inpainting)")

    # Create LAYOUT CONTROL frames (trajectory guidance for chunks 1+)
    # NOT the subject image - that comes from extension mode on chunk 0
    print("\nCreating layout control frames...")
    layout_frames = create_layout_control_frames(
        positions=positions,
        width=config["width"],
        height=config["height"],
        mask_radius=config["layout_radius"],
    )
    print(f"  Layout frames shape: {layout_frames.shape}")
    print(
        f"  First frame mean: {layout_frames[0].mean():.1f} (white bg with black contour)"
    )

    # Save intermediate visualizations
    if config["save_intermediates"]:
        print("\nSaving intermediate videos...")

        # Layout control frames (the actual VACE control signal for chunks 1+)
        # Note: trajectory.mp4 was redundant - this IS the trajectory visualization
        # in the format the model actually sees (white bg + black contour)
        export_to_video(
            layout_frames / 255.0, output_dir / "layout_control.mp4", fps=16
        )
        print("  Saved: layout_control.mp4")

    # Initialize pipeline
    print("\nInitializing LongLive pipeline with VACE...")

    vace_path = str(
        get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
    )

    # Build LoRA list from config
    loras = []
    if "loras" in config and config["loras"]:
        for lora_path in config["loras"]:
            if lora_path:
                loras.append({"path": lora_path, "scale": 1.0})

    pipeline_config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
            "loras": loras if loras else [],
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
            "vae_type": config["vae_type"],
        }
    )

    # VACE in_dim = 96 for masked/inpainting mode (32 latent + 64 mask encoding)
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline ready\n")

    # Preprocess for VACE (layout control for chunks 1+)
    vace_input_tensor = preprocess_frames_for_vace(layout_frames, device)
    vace_mask_tensor = preprocess_masks_for_vace(masks, device)

    print(
        f"Layout tensor: {vace_input_tensor.shape}, range [{vace_input_tensor.min():.2f}, {vace_input_tensor.max():.2f}]"
    )
    print(
        f"Mask tensor: {vace_mask_tensor.shape}, range [{vace_mask_tensor.min():.2f}, {vace_mask_tensor.max():.2f}]"
    )

    # Generate video
    print("\n" + "=" * 40)
    print("  GENERATING VIDEO")
    print("=" * 40)

    outputs = []
    latency_measures = []
    frames_per_chunk = config["frames_per_chunk"]

    for chunk_idx in range(config["num_chunks"]):
        start_time = time.time()

        start_frame = chunk_idx * frames_per_chunk
        end_frame = start_frame + frames_per_chunk

        # Extract chunk tensors
        input_chunk = vace_input_tensor[:, :, start_frame:end_frame, :, :]
        mask_chunk = vace_mask_tensor[:, :, start_frame:end_frame, :, :]

        # Chunk 0: Extension mode ONLY - establishes subject identity
        # Chunks 1+: Trajectory control - subject identity from KV cache
        if chunk_idx == 0:
            kwargs = {
                "prompts": [{"text": config["prompt"], "weight": 100}],
                "first_frame_image": str(subject_path),
            }
            mode_str = "extension"
        else:
            kwargs = {
                "prompts": [{"text": config["prompt"], "weight": 100}],
                "vace_context_scale": config["vace_context_scale"],
                "vace_input_frames": input_chunk,
                "vace_input_masks": mask_chunk,
            }
            mode_str = "trajectory"

        print(f"Chunk {chunk_idx}: frames {start_frame}-{end_frame} ({mode_str})")

        # Generate
        output = pipeline(**kwargs)

        latency = time.time() - start_time
        fps = output.shape[0] / latency
        print(f"  -> {output.shape[0]} frames, {latency:.1f}s, {fps:.1f} fps")

        latency_measures.append(latency)

        # Convert to numpy uint8 and apply FPS overlay
        output_np = output.detach().cpu().numpy()
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        output_np = add_fps_overlay(output_np, fps, chunk_idx, config["num_chunks"])
        outputs.append(output_np)

    # Concatenate outputs (now numpy arrays with FPS overlay)
    output_video = np.concatenate(outputs, axis=0)
    print(f"\nFinal output: {output_video.shape}")

    # Save output (already numpy uint8 with FPS overlay)
    output_video_np = output_video.astype(np.float32) / 255.0

    output_filename = f"output_{pattern}.mp4"
    output_path = output_dir / output_filename
    export_to_video(output_video_np, output_path, fps=16)

    print(f"\nSaved: {output_path}")

    # Concatenate layout control video with output video side-by-side
    if config["save_intermediates"]:
        print("\nConcatenating layout control with output...")
        layout_path = output_dir / "layout_control.mp4"
        concat_path = output_dir / f"output_{pattern}_with_control.mp4"

        # Read both videos and concatenate
        cap_layout = cv2.VideoCapture(str(layout_path))
        cap_output = cv2.VideoCapture(str(output_path))

        concat_frames = []
        while True:
            ret1, frame1 = cap_layout.read()
            ret2, frame2 = cap_output.read()
            if not ret1 or not ret2:
                break
            # BGR to RGB
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            concat_frames.append(np.hstack([frame1, frame2]))

        cap_layout.release()
        cap_output.release()

        if concat_frames:
            concat_array = np.array(concat_frames) / 255.0
            export_to_video(concat_array, concat_path, fps=16)
            print(f"  Saved: {concat_path}")

    # Stats
    total_time = sum(latency_measures)
    total_frames = output_video.shape[0]
    avg_fps = total_frames / total_time
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Total frames: {total_frames}")
    print(f"Average FPS: {avg_fps:.2f}")

    print("\n" + "=" * 80)
    print("  COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    if config["save_intermediates"]:
        print("Intermediates:")
        print("  - layout_control.mp4: Layout control signal (white bg, black contour)")
        print(f"  - output_{pattern}_with_control.mp4: Side-by-side comparison")


if __name__ == "__main__":
    main()
