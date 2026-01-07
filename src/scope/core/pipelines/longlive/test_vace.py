"""
Unified test script for LongLive pipeline with VACE integration.

Supports multiple modes:
- R2V: Reference-to-Video generation using reference images
- Depth guidance: Structural guidance using depth maps
- Inpainting: Masked video-to-video generation

Modes can be combined:
- R2V + Depth
- R2V + Inpainting
- Depth only
- Inpainting only
- R2V only

Usage:
    Edit the CONFIG dictionary below to enable/disable modes and set paths.
    python -m scope.core.pipelines.longlive.test_vace
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from ..video import load_video
from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== MODE SELECTION =====
    "use_r2v": False,  # Reference-to-Video: condition on reference images
    "use_depth": False,  # Depth guidance: structural control via depth maps
    "use_inpainting": True,  # Inpainting: masked video-to-video generation
    # ===== INPUT PATHS =====
    # R2V: List of reference image paths
    "ref_images": [
        "frontend/public/assets/example.png",  # path/to/image.png
    ],
    # Depth: Path to depth map video (grayscale or RGB, will be converted)
    "depth_video": "vace_tests/control_frames_depth.mp4",  # path/to/depth_video.mp4
    # Inpainting: Input video and mask video paths
    "input_video": "frontend/public/assets/test.mp4",  # path/to/input_video.mp4
    "mask_video": "vace_tests/circle_mask.mp4",  # path/to/mask_video.mp4
    # ===== GENERATION PARAMETERS =====
    "prompt": None,  # Set to override mode-specific prompts, or None to use defaults
    "prompt_r2v": "",  # Default prompt for R2V mode
    "prompt_depth": "a cat walking towards the camera",  # Default prompt for depth mode
    "prompt_inpainting": "a fireball",  # Default prompt for inpainting mode
    "num_chunks": 3,  # Number of generation chunks
    "frames_per_chunk": 12,  # Frames per chunk (12 = 3 latent * 4 temporal upsample)
    "height": 512,
    "width": 512,
    "vace_context_scale": 0.7,  # VACE conditioning strength (0.0-1.0)
    # ===== INPAINTING SPECIFIC =====
    "mask_threshold": 0.5,  # Threshold for binarizing mask (0-1)
    "mask_value": 127,  # Gray value for masked regions (0-255)
    # ===== OUTPUT =====
    "output_dir": "vace_tests/unified",  # path/to/output_dir
    "vae_type": "tae",
}

# ========================= END CONFIGURATION =========================

# ============================= UTILITIES =============================


def preprocess_depth_frames(
    depth_frames: torch.Tensor,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess depth frames for VACE depth guidance.

    Args:
        depth_frames: Depth frames [F, H, W] in [0, 1]
        target_height: Target height
        target_width: Target width
        device: Target device

    Returns:
        Preprocessed depth tensor [1, 3, F, H, W] in [-1, 1]
    """
    num_frames, orig_height, orig_width = depth_frames.shape

    # Resize if needed
    if orig_height != target_height or orig_width != target_width:
        depth_frames = depth_frames.unsqueeze(1)
        depth_frames = torch.nn.functional.interpolate(
            depth_frames,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        depth_frames = depth_frames.squeeze(1)

    # Convert single-channel depth to 3-channel RGB (replicate across channels)
    depth_frames = depth_frames.unsqueeze(1).repeat(1, 3, 1, 1)

    # Normalize to [-1, 1] for VAE encoding
    depth_frames = depth_frames * 2.0 - 1.0

    # Add batch dimension and rearrange to [1, 3, F, H, W]
    depth_frames = depth_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return depth_frames.to(device)


def extract_depth_chunk(
    depth_video: torch.Tensor,
    chunk_index: int,
    frames_per_chunk: int,
) -> torch.Tensor:
    """
    Extract a chunk from depth video tensor.

    Args:
        depth_video: Depth video tensor [1, 3, F, H, W]
        chunk_index: Chunk index
        frames_per_chunk: Number of frames per chunk

    Returns:
        Chunk tensor [1, 3, frames_per_chunk, H, W]
    """
    start_idx = chunk_index * frames_per_chunk
    end_idx = start_idx + frames_per_chunk

    # Clamp to video length
    total_frames = depth_video.shape[2]
    end_idx = min(end_idx, total_frames)

    chunk = depth_video[:, :, start_idx:end_idx, :, :]

    # Pad if needed
    if chunk.shape[2] < frames_per_chunk:
        padding = frames_per_chunk - chunk.shape[2]
        pad_frames = chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
        chunk = torch.cat([chunk, pad_frames], dim=2)

    return chunk


def resolve_path(path_str: str, relative_to: Path) -> Path:
    """Resolve path relative to a base directory or as absolute."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (relative_to / path).resolve()


def load_video_frames(
    video_path: Path,
    target_height: int,
    target_width: int,
    max_frames: int = None,
) -> np.ndarray:
    """
    Load video frames from file.

    Args:
        video_path: Path to video file
        target_height: Target height for resizing
        target_width: Target width for resizing
        max_frames: Maximum number of frames to load (None = all frames)

    Returns:
        Numpy array of shape [F, H, W, C] with values in [0, 255]
    """
    print(f"load_video_frames: Loading {video_path}")

    # Use load_video which returns [C, T, H, W] tensor in [0, 255] when normalize=False
    video_tensor = load_video(
        str(video_path),
        num_frames=max_frames,
        resize_hw=(target_height, target_width),
        normalize=False,
    )

    # Convert from [C, T, H, W] to [F, H, W, C]
    video_tensor = video_tensor.permute(1, 2, 3, 0)

    # Convert to numpy uint8
    frames_array = video_tensor.numpy().astype(np.uint8)

    num_frames = frames_array.shape[0]
    print(
        f"load_video_frames: Loaded {num_frames} frames at {target_height}x{target_width}"
    )
    return frames_array


def create_mask_from_video(
    mask_video_path: Path,
    num_frames: int,
    threshold: float,
) -> np.ndarray:
    """
    Create binary mask from video frames.

    Args:
        mask_video_path: Path to mask video file
        num_frames: Number of frames needed
        threshold: Threshold for binarization (0-1)

    Returns:
        Binary mask array of shape [F, H, W] with values in {0, 1}
    """
    print(f"create_mask_from_video: Loading {mask_video_path}")

    # Load mask video (will be resized by load_video_frames)
    mask_frames = load_video_frames(
        mask_video_path,
        target_height=512,  # Will be properly resized later
        target_width=512,
        max_frames=None,
    )

    # Convert to grayscale and normalize
    mask_gray = np.mean(mask_frames, axis=-1) / 255.0

    # Binarize
    binary_mask = (mask_gray > threshold).astype(np.float32)

    # Handle length mismatch with looping or truncation
    mask_frames_count = binary_mask.shape[0]
    if mask_frames_count < num_frames:
        print(
            f"create_mask_from_video: Looping mask from {mask_frames_count} to {num_frames} frames"
        )
        repeats_needed = (num_frames // mask_frames_count) + 1
        binary_mask = np.tile(binary_mask, (repeats_needed, 1, 1))[:num_frames]
    elif mask_frames_count > num_frames:
        print(
            f"create_mask_from_video: Truncating mask from {mask_frames_count} to {num_frames} frames"
        )
        binary_mask = binary_mask[:num_frames]

    print(
        f"create_mask_from_video: Mask shape={binary_mask.shape}, "
        f"range=[{binary_mask.min():.2f}, {binary_mask.max():.2f}], "
        f"mean={binary_mask.mean():.3f}"
    )
    return binary_mask


def create_masked_video(
    video_frames: np.ndarray,
    mask: np.ndarray,
    mask_value: int,
) -> np.ndarray:
    """
    Apply mask to video by filling masked regions with gray value.

    Args:
        video_frames: Original video [F, H, W, C] in [0, 255]
        mask: Binary mask [F, H, W] in {0, 1} (1=inpaint, 0=preserve)
        mask_value: Gray value for masked regions

    Returns:
        Masked video [F, H, W, C] in [0, 255]
    """
    print(f"create_masked_video: Applying mask with value={mask_value}")

    mask_expanded = mask[..., np.newaxis]
    masked_video = np.where(
        mask_expanded > 0.5,
        mask_value,
        video_frames,
    ).astype(np.uint8)

    return masked_video


def preprocess_video_for_vace(
    video_frames: np.ndarray,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess video frames for VACE input.

    Args:
        video_frames: Video frames [F, H, W, C] in [0, 255]
        target_height: Target height
        target_width: Target width
        device: Target device

    Returns:
        Preprocessed tensor [1, 3, F, H, W] in [-1, 1]
    """
    video_tensor = torch.from_numpy(video_frames).float() / 255.0

    # Resize if needed
    num_frames, orig_height, orig_width, channels = video_tensor.shape
    if orig_height != target_height or orig_width != target_width:
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        video_tensor = video_tensor.permute(0, 3, 1, 2)

    # Normalize to [-1, 1] for VAE
    video_tensor = video_tensor * 2.0 - 1.0

    # Add batch dimension: [1, C, F, H, W]
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return video_tensor.to(device)


def preprocess_mask(
    mask: np.ndarray,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess mask for VACE input.

    Args:
        mask: Binary mask [F, H, W] in {0, 1}
        target_height: Target height
        target_width: Target width
        device: Target device

    Returns:
        Preprocessed mask [1, 1, F, H, W]
    """
    mask_tensor = torch.from_numpy(mask).float()

    # Resize if needed
    num_frames, orig_height, orig_width = mask_tensor.shape
    if orig_height != target_height or orig_width != target_width:
        mask_tensor = mask_tensor.unsqueeze(1)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(target_height, target_width),
            mode="nearest",
        )
    else:
        mask_tensor = mask_tensor.unsqueeze(1)

    # Add batch dimension: [1, 1, F, H, W]
    mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

    return mask_tensor.to(device)


def load_depth_video(
    depth_video_path: Path,
    target_height: int,
    target_width: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and preprocess depth video.

    Args:
        depth_video_path: Path to depth video
        target_height: Target height
        target_width: Target width
        max_frames: Maximum frames to load
        device: Target device

    Returns:
        Depth tensor [1, 3, F, H, W] preprocessed for VACE
    """
    print(f"load_depth_video: Loading {depth_video_path}")

    depth_frames_rgb = load_video_frames(
        depth_video_path,
        target_height=target_height,
        target_width=target_width,
        max_frames=max_frames,
    )

    # Convert RGB to grayscale (take first channel)
    depth_frames_np = depth_frames_rgb[:, :, :, 0].astype(np.float32) / 255.0

    # Preprocess with vace_utils function
    depth_frames_tensor = torch.from_numpy(depth_frames_np).float()
    depth_video = preprocess_depth_frames(
        depth_frames_tensor,
        target_height,
        target_width,
        device,
    )

    print(f"load_depth_video: Depth shape={depth_video.shape}")
    return depth_video


# ============================= MAIN =============================


def main():
    print("=" * 80)
    print("  LongLive Unified Test Script")
    print("=" * 80)

    # Parse configuration
    config = CONFIG
    use_r2v = config["use_r2v"]
    use_depth = config["use_depth"]
    use_inpainting = config["use_inpainting"]

    # Validate mode selection
    if use_depth and use_inpainting:
        raise ValueError("Cannot use both depth and inpainting modes simultaneously")

    if not (use_r2v or use_depth or use_inpainting):
        raise ValueError("At least one mode must be enabled")

    # Select appropriate prompt based on mode
    if config["prompt"] is not None:
        # User override
        prompt = config["prompt"]
    elif use_inpainting:
        # Inpainting takes priority
        prompt = config["prompt_inpainting"]
    elif use_depth:
        # Depth guidance
        prompt = config["prompt_depth"]
    else:
        # R2V only
        prompt = config["prompt_r2v"]

    print("\nConfiguration:")
    print(f"  R2V: {use_r2v}")
    print(f"  Depth Guidance: {use_depth}")
    print(f"  Inpainting: {use_inpainting}")
    print(f"  Prompt: '{prompt}'")
    print(f"  Chunks: {config['num_chunks']} x {config['frames_per_chunk']} frames")
    print(f"  Resolution: {config['height']}x{config['width']}")
    print(f"  VACE Scale: {config['vace_context_scale']}")

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
            "vace_path": vace_path
            if (use_r2v or use_depth or use_inpainting)
            else None,
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

    # Set vace_in_dim for depth/inpainting modes
    if use_depth or use_inpainting:
        pipeline_config.model_config.base_model_kwargs = (
            pipeline_config.model_config.base_model_kwargs or {}
        )
        pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline ready\n")

    # Prepare inputs
    total_frames = config["num_chunks"] * config["frames_per_chunk"]

    ref_images = None
    depth_video = None
    input_video_tensor = None
    mask_tensor = None
    input_frames_np = None
    mask_np = None

    # Load reference images for R2V
    if use_r2v:
        print("=== Preparing R2V Inputs ===")
        ref_image_paths = []
        for img_path in config["ref_images"]:
            resolved = resolve_path(img_path, project_root)
            if resolved.exists():
                ref_image_paths.append(str(resolved))
                print(f"  Reference image: {resolved}")
            else:
                print(f"  Warning: Reference image not found: {resolved}")

        if ref_image_paths:
            ref_images = ref_image_paths
        else:
            print("  No valid reference images found, disabling R2V")
            use_r2v = False
        print()

    # Load depth video
    if use_depth:
        print("=== Preparing Depth Inputs ===")
        depth_video_path = resolve_path(config["depth_video"], script_dir)
        if not depth_video_path.exists():
            raise FileNotFoundError(f"Depth video not found: {depth_video_path}")

        depth_video = load_depth_video(
            depth_video_path,
            config["height"],
            config["width"],
            total_frames,
            device,
        )
        print()

    # Load inpainting inputs
    if use_inpainting:
        print("=== Preparing Inpainting Inputs ===")

        # Load input video
        input_video_path = resolve_path(config["input_video"], project_root)
        if not input_video_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        input_frames_np = load_video_frames(
            input_video_path,
            config["height"],
            config["width"],
            max_frames=total_frames,
        )

        # Load mask video
        mask_video_path = resolve_path(config["mask_video"], script_dir)
        if not mask_video_path.exists():
            raise FileNotFoundError(f"Mask video not found: {mask_video_path}")

        mask_np = create_mask_from_video(
            mask_video_path,
            total_frames,
            config["mask_threshold"],
        )

        # Create masked video
        masked_frames = create_masked_video(
            input_frames_np,
            mask_np,
            config["mask_value"],
        )

        # Preprocess
        input_video_tensor = preprocess_video_for_vace(
            masked_frames,
            config["height"],
            config["width"],
            device,
        )
        mask_tensor = preprocess_mask(
            mask_np,
            config["height"],
            config["width"],
            device,
        )

        # Save masked video preview
        masked_preview_path = output_dir / "input_masked_video.mp4"
        export_to_video(
            masked_frames.astype(np.float32) / 255.0,
            masked_preview_path,
            fps=16,
        )
        print(f"  Saved masked preview: {masked_preview_path}")
        print()

    # Generate video
    print("=== Generating Video ===")
    outputs = []
    latency_measures = []
    fps_measures = []

    frames_per_chunk = config["frames_per_chunk"]
    overlap_frames = 3 if use_inpainting else 0
    start_idx = 0
    is_first_chunk = True

    for chunk_index in range(config["num_chunks"]):
        start_time = time.time()

        # Calculate chunk boundaries
        if use_inpainting and chunk_index > 0:
            start_idx = start_idx + frames_per_chunk - overlap_frames
        elif not use_inpainting:
            start_idx = chunk_index * frames_per_chunk

        end_idx = start_idx + frames_per_chunk

        # Prepare pipeline kwargs
        kwargs = {
            "prompts": [{"text": prompt, "weight": 100}],
            "vace_context_scale": config["vace_context_scale"],
        }

        # Add R2V reference images (first chunk only)
        if use_r2v and is_first_chunk and ref_images:
            kwargs["vace_ref_images"] = ref_images
            print(f"Chunk {chunk_index}: Using {len(ref_images)} reference image(s)")

        # Add depth guidance
        if use_depth:
            depth_frames_available = depth_video.shape[2]
            depth_frames_needed = end_idx

            if depth_frames_needed > depth_frames_available:
                print(f"Chunk {chunk_index}: Not enough depth frames, stopping")
                break

            depth_chunk = extract_depth_chunk(
                depth_video,
                chunk_index,
                frames_per_chunk,
            )
            kwargs["vace_input_frames"] = depth_chunk
            print(f"Chunk {chunk_index}: Depth chunk shape={depth_chunk.shape}")

        # Add inpainting inputs
        if use_inpainting:
            total_frames_available = input_video_tensor.shape[2]
            end_idx_clamped = min(end_idx, total_frames_available)

            input_chunk = input_video_tensor[:, :, start_idx:end_idx_clamped, :, :]
            mask_chunk = mask_tensor[:, :, start_idx:end_idx_clamped, :, :]

            # Pad last chunk if needed
            if input_chunk.shape[2] < frames_per_chunk:
                padding = frames_per_chunk - input_chunk.shape[2]
                input_pad = input_chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
                mask_pad = mask_chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
                input_chunk = torch.cat([input_chunk, input_pad], dim=2)
                mask_chunk = torch.cat([mask_chunk, mask_pad], dim=2)

            kwargs["vace_input_frames"] = input_chunk
            kwargs["vace_input_masks"] = mask_chunk

            print(
                f"Chunk {chunk_index}: start={start_idx}, end={end_idx_clamped}, "
                f"overlap={overlap_frames if chunk_index > 0 else 0}, "
                f"input_shape={input_chunk.shape}"
            )

        # Generate
        output = pipeline(**kwargs)
        is_first_chunk = False

        # Metrics
        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start_time
        fps = num_output_frames / latency

        print(
            f"Chunk {chunk_index}: Generated {num_output_frames} frames, "
            f"latency={latency:.2f}s, fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs.append(output.detach().cpu())

    # Concatenate outputs
    if use_inpainting and overlap_frames > 0:
        output_chunks = []
        for chunk_idx, output_chunk in enumerate(outputs):
            if chunk_idx == 0:
                output_chunks.append(output_chunk)
            else:
                output_chunks.append(output_chunk[overlap_frames:])
        output_video = torch.concat(output_chunks)
    else:
        output_video = torch.concat(outputs)

    print(f"\nFinal output shape: {output_video.shape}")

    # Save output video
    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    mode_suffix = []
    if use_r2v:
        mode_suffix.append("r2v")
    if use_depth:
        mode_suffix.append("depth")
    if use_inpainting:
        mode_suffix.append("inpainting")

    output_filename = f"output_{'_'.join(mode_suffix)}.mp4"
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

    # Save additional visualizations
    if use_depth and depth_video is not None:
        print("\n=== Saving Depth Visualization ===")
        depth_vis_frames = output_video.shape[0]
        depth_vis = depth_video[0, 0, :depth_vis_frames].cpu().numpy()
        depth_vis = ((depth_vis + 1.0) / 2.0 * 255).astype(np.uint8)
        depth_vis_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
        export_to_video(
            depth_vis_rgb / 255.0,
            output_dir / "depth_maps.mp4",
            fps=16,
        )
        print(f"  Saved: {output_dir / 'depth_maps.mp4'}")

    if use_inpainting and mask_np is not None:
        print("\n=== Saving Inpainting Visualizations ===")

        # Mask visualization
        mask_viz_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
        export_to_video(
            mask_viz_rgb,
            output_dir / "mask_visualization.mp4",
            fps=16,
        )
        print(f"  Saved: {output_dir / 'mask_visualization.mp4'}")

        # Original video
        original_resized = torch.from_numpy(input_frames_np).float() / 255.0
        original_resized = original_resized.permute(0, 3, 1, 2)
        original_resized = torch.nn.functional.interpolate(
            original_resized,
            size=(config["height"], config["width"]),
            mode="bilinear",
            align_corners=False,
        )
        original_resized = original_resized.permute(0, 2, 3, 1).numpy()
        export_to_video(
            original_resized,
            output_dir / "input_original.mp4",
            fps=16,
        )
        print(f"  Saved: {output_dir / 'input_original.mp4'}")

    print("\n" + "=" * 80)
    print("  Test Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Main output: {output_filename}")
    if use_r2v and ref_images:
        print(f"Used {len(ref_images)} reference image(s) for R2V conditioning")
    if use_depth:
        print("Used depth maps for structural guidance")
    if use_inpainting:
        print("Used spatial masks for inpainting control")


if __name__ == "__main__":
    main()
