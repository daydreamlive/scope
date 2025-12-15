"""
Test script combining VACE R2V (Reference-to-Video) with inpainting functionality.

This script demonstrates:
1. Reference image conditioning (R2V) - condition generation on reference images
2. Inpainting - masked video-to-video generation with spatial control
3. Combined mode - use both reference images and inpainting masks together

Features:
- Load input video and mask video for inpainting
- Optionally use reference images for style/content conditioning
- Generate video with both VACE conditioning and spatial mask control
- Process video in chunks with overlap for smooth transitions

Usage:
    python -m scope.core.pipelines.longlive.test_vace_inpainting
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline


def load_video_frames(video_path: str, max_frames: int = None) -> np.ndarray:
    """
    Load video frames from file.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to load (None = all frames)

    Returns:
        Numpy array of shape [F, H, W, C] with values in [0, 255]
    """
    print(f"load_video_frames: Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"load_video_frames: Failed to open video {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"load_video_frames: No frames loaded from {video_path}")

    frames_array = np.stack(frames, axis=0)
    print(
        f"load_video_frames: Loaded {len(frames)} frames with shape {frames_array.shape}"
    )
    return frames_array


def create_mask_from_video(
    mask_video_path: str, num_frames: int, threshold: float = 0.5
) -> np.ndarray:
    """
    Create binary mask from video frames.

    Args:
        mask_video_path: Path to mask video file
        num_frames: Number of frames to extract
        threshold: Threshold for binarization (0-1), values above threshold become white (1)

    Returns:
        Binary mask array of shape [F, H, W] with values in {0, 1}
    """
    print(f"create_mask_from_video: Loading mask video from {mask_video_path}")

    # Load mask video frames
    mask_frames = load_video_frames(mask_video_path, max_frames=num_frames)

    # Convert to grayscale
    mask_gray = np.mean(mask_frames, axis=-1)

    # Normalize to [0, 1]
    mask_normalized = mask_gray / 255.0

    # Threshold to create binary mask (white=1=generate, black=0=preserve)
    binary_mask = (mask_normalized > threshold).astype(np.float32)

    print(f"create_mask_from_video: Created binary mask with shape {binary_mask.shape}")
    print(
        f"create_mask_from_video: Mask stats - min: {binary_mask.min()}, max: {binary_mask.max()}, mean: {binary_mask.mean():.3f}"
    )

    return binary_mask


def create_masked_video(
    video_frames: np.ndarray, mask: np.ndarray, mask_value: int = 127
) -> np.ndarray:
    """
    Create masked video by filling masked regions with gray value.

    Args:
        video_frames: Original video frames [F, H, W, C] in [0, 255]
        mask: Binary mask [F, H, W] with values in {0, 1} (1=inpaint, 0=preserve)
        mask_value: Gray value to fill masked regions (default 127 for middle gray)

    Returns:
        Masked video frames [F, H, W, C] in [0, 255]
    """
    print(f"create_masked_video: Creating masked video (mask_value={mask_value})")

    # Expand mask to match video channels
    mask_expanded = mask[..., np.newaxis]  # [F, H, W, 1]

    # Create masked video: where mask=1 (inpaint), set to gray; where mask=0 (preserve), keep original
    masked_video = video_frames.copy()
    masked_video = np.where(
        mask_expanded > 0.5,
        mask_value,  # Masked region (to be inpainted)
        video_frames,  # Preserved region
    ).astype(np.uint8)

    print(f"create_masked_video: Created masked video with shape {masked_video.shape}")

    return masked_video


def preprocess_video_for_inpainting(
    video_frames: np.ndarray,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess video frames for VACE inpainting input.

    Args:
        video_frames: Video frames [F, H, W, C] in range [0, 255]
        target_height: Target height
        target_width: Target width
        device: Target device

    Returns:
        Preprocessed tensor [1, 3, F, H, W] in range [-1, 1] (matching VAE expectation)
    """
    print(f"preprocess_video_for_inpainting: Input shape {video_frames.shape}")

    # Convert to tensor and normalize to [0, 1]
    video_tensor = torch.from_numpy(video_frames).float() / 255.0

    # Resize if needed
    num_frames, orig_height, orig_width, channels = video_tensor.shape
    if orig_height != target_height or orig_width != target_width:
        print(
            f"preprocess_video_for_inpainting: Resizing from {orig_height}x{orig_width} to {target_height}x{target_width}"
        )
        # Rearrange to [F, C, H, W] for interpolation
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        # Back to [F, C, H, W]
    else:
        video_tensor = video_tensor.permute(0, 3, 1, 2)

    # Normalize to [-1, 1] for VAE encoding (matching preprocess_depth_frames)
    video_tensor = video_tensor * 2.0 - 1.0

    # Add batch dimension: [1, C, F, H, W]
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

    # Move to device
    video_tensor = video_tensor.to(device)

    print(f"preprocess_video_for_inpainting: Output shape {video_tensor.shape}")
    return video_tensor


def preprocess_mask(
    mask: np.ndarray,
    target_height: int,
    target_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Preprocess mask for VACE input.

    Args:
        mask: Binary mask [F, H, W] with values in {0, 1}
        target_height: Target height
        target_width: Target width
        device: Target device

    Returns:
        Preprocessed mask tensor [1, 1, F, H, W]
    """
    print(f"preprocess_mask: Input shape {mask.shape}")

    # Convert to tensor
    mask_tensor = torch.from_numpy(mask).float()

    # Resize if needed
    num_frames, orig_height, orig_width = mask_tensor.shape
    if orig_height != target_height or orig_width != target_width:
        print(
            f"preprocess_mask: Resizing from {orig_height}x{orig_width} to {target_height}x{target_width}"
        )
        # Add channel dim [F, 1, H, W] for interpolation
        mask_tensor = mask_tensor.unsqueeze(1)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(target_height, target_width), mode="nearest"
        )
        # Back to [F, 1, H, W]
    else:
        mask_tensor = mask_tensor.unsqueeze(1)

    # Add batch dimension: [1, 1, F, H, W]
    mask_tensor = mask_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)

    # Move to device
    mask_tensor = mask_tensor.to(device)

    print(f"preprocess_mask: Output shape {mask_tensor.shape}")
    return mask_tensor


def main():
    print("=== LongLive VACE + Inpainting Combined Test ===\n")

    # Paths
    # __file__ is in src/scope/core/pipelines/longlive/
    # Go up 6 levels to get to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent

    # Configuration flags
    USE_INPAINTING = True  # Set to False to disable inpainting (pure R2V)
    USE_REF_IMAGES = True  # Set to False to disable reference images (pure inpainting)

    # Input paths
    input_video_path = project_root / "frontend" / "public" / "assets" / "test.mp4"
    mask_video_path = (
        project_root
        / "src"
        / "scope"
        / "core"
        / "pipelines"
        / "longlive"
        / "vace_tests"
        / "static_mask_half_white_half_black.mp4"
    )
    ref_images = [
        str(project_root / "example.png")
    ]  # Reference images for R2V conditioning

    output_dir = (
        project_root
        / "src"
        / "scope"
        / "core"
        / "pipelines"
        / "longlive"
        / "vace_tests"
        / "vace_r2v_inpainting"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Configuration:")
    print(f"  USE_INPAINTING: {USE_INPAINTING}")
    print(f"  USE_REF_IMAGES: {USE_REF_IMAGES}")
    print(f"Input video: {input_video_path}")
    if USE_INPAINTING:
        print(f"Mask video: {mask_video_path}")
    if USE_REF_IMAGES:
        print(f"Reference images: {ref_images}")
    print(f"Output dir: {output_dir}\n")

    # Check if reference images exist
    if USE_REF_IMAGES:
        existing_ref_images = [img for img in ref_images if Path(img).exists()]
        if not existing_ref_images:
            print("Warning: No reference images found. Disabling R2V conditioning.")
            USE_REF_IMAGES = False
        else:
            ref_images = existing_ref_images
            print(f"Using {len(ref_images)} reference image(s) for R2V conditioning")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Check if VACE model is available
    vace_model_path = Path.home() / ".daydream-scope" / "models" / "Wan2.1-VACE-1.3B"
    vace_checkpoint = vace_model_path / "diffusion_pytorch_model.safetensors"

    if not vace_checkpoint.exists():
        print(f"VACE checkpoint not found at {vace_checkpoint}")
        print(
            "Please download VACE weights to ~/.daydream-scope/models/Wan2.1-VACE-1.3B/"
        )
        if USE_INPAINTING:
            print("Falling back to inpainting mode (VACE required for R2V)")
        vace_path = None
    else:
        vace_path = str(vace_checkpoint)
        print(f"Using VACE model from {vace_path}")

    config = OmegaConf.create(
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
            "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
            "height": 512,
            "width": 512,
        }
    )

    # Override vace_in_dim for inpainting mode
    if USE_INPAINTING:
        config.model_config.base_model_kwargs = (
            config.model_config.base_model_kwargs or {}
        )
        config.model_config.base_model_kwargs["vace_in_dim"] = (
            96  # Use 96 to load pretrained VACE weights
        )

    # Load pipeline
    print("Loading pipeline...")
    pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
    print("Pipeline loaded\n")

    # Parameters
    prompt_text = "A girl, high quality, cinematic"
    num_chunks = 3
    frames_per_chunk = 12
    total_frames = num_chunks * frames_per_chunk

    print(f"Prompt: {prompt_text}")
    print(
        f"Generating {num_chunks} chunks, {frames_per_chunk} frames per chunk = {total_frames} total frames\n"
    )

    # Load input video (if using inpainting)
    input_video_tensor = None
    mask_tensor = None
    input_frames = None
    mask = None

    if USE_INPAINTING:
        print("=== Loading Input Video ===")
        input_frames = load_video_frames(str(input_video_path), max_frames=total_frames)
        print()

        # Create mask from mask video
        print("=== Creating Mask ===")
        mask = create_mask_from_video(
            str(mask_video_path), num_frames=total_frames, threshold=0.5
        )
        print()

        # Create masked video (gray out regions to be inpainted)
        print("=== Creating Masked Video ===")
        masked_frames = create_masked_video(input_frames, mask, mask_value=127)

        # Save masked video for visualization
        masked_video_path = output_dir / "input_masked_video.mp4"
        masked_frames_normalized = masked_frames.astype(np.float32) / 255.0
        export_to_video(masked_frames_normalized, masked_video_path, fps=16)
        print(f"Saved masked video to {masked_video_path}\n")

        # Preprocess inputs
        print("=== Preprocessing Inputs ===")
        input_video_tensor = preprocess_video_for_inpainting(
            masked_frames,
            config.height,
            config.width,
            device,
        )
        mask_tensor = preprocess_mask(
            mask,
            config.height,
            config.width,
            device,
        )
        print()

    # Generate with combined VACE + inpainting
    print("=== Generating Video ===")
    outputs = []
    latency_measures = []
    fps_measures = []

    # Use overlapping chunks for smooth transitions (LongLive expects 3-frame overlap)
    overlap_frames = 3 if USE_INPAINTING else 0
    start_idx = 0

    for chunk_index in range(num_chunks):
        start = time.time()

        # Calculate chunk boundaries
        if USE_INPAINTING and chunk_index > 0:
            # Start 3 frames before previous chunk ended (overlap)
            start_idx = start_idx + frames_per_chunk - overlap_frames
        elif not USE_INPAINTING:
            # No overlap for pure R2V mode
            start_idx = chunk_index * frames_per_chunk

        end_idx = start_idx + frames_per_chunk

        # Prepare pipeline kwargs
        kwargs = {
            "prompts": [{"text": prompt_text, "weight": 100}],
            "vace_context_scale": 1.0,
        }

        # Add reference images (only on first chunk for R2V)
        if USE_REF_IMAGES and chunk_index == 0 and vace_path is not None:
            kwargs["ref_images"] = ref_images
            print(f"Chunk {chunk_index}: Using {len(ref_images)} reference image(s)")

        # Add inpainting inputs (if enabled)
        if USE_INPAINTING:
            # Extract overlapping chunks
            total_frames = input_video_tensor.shape[2]
            end_idx_clamped = min(end_idx, total_frames)

            input_chunk = input_video_tensor[:, :, start_idx:end_idx_clamped, :, :]
            mask_chunk = mask_tensor[:, :, start_idx:end_idx_clamped, :, :]

            # Pad if needed (for last chunk)
            if input_chunk.shape[2] < frames_per_chunk:
                padding = frames_per_chunk - input_chunk.shape[2]
                input_pad = input_chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
                mask_pad = mask_chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
                input_chunk = torch.cat([input_chunk, input_pad], dim=2)
                mask_chunk = torch.cat([mask_chunk, mask_pad], dim=2)

            kwargs["input_frames"] = input_chunk
            kwargs["input_masks"] = mask_chunk

            print(
                f"Chunk {chunk_index}: start_idx={start_idx}, end_idx={end_idx_clamped}, "
                f"overlap={overlap_frames if chunk_index > 0 else 0}, "
                f"input_chunk shape={input_chunk.shape}, mask_chunk shape={mask_chunk.shape}"
            )

        # Generate
        output = pipeline(**kwargs)

        num_output_frames_chunk, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames_chunk / latency

        print(
            f"Chunk {chunk_index}: Generated {num_output_frames_chunk} frames "
            f"latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs.append(output.detach().cpu())

    # Concatenate chunks, skipping overlapping frames from subsequent chunks
    if USE_INPAINTING:
        output_chunks = []
        for chunk_idx, output_chunk in enumerate(outputs):
            if chunk_idx == 0:
                # First chunk: take all frames
                output_chunks.append(output_chunk)
            else:
                # Subsequent chunks: skip overlapping frames
                output_chunks.append(output_chunk[overlap_frames:])
        output_video = torch.concat(output_chunks)
    else:
        # No overlap, just concatenate
        output_video = torch.concat(outputs)

    print(f"\nFinal output shape: {output_video.shape}")
    output_video_np = output_video.contiguous().numpy()
    # Ensure values are in [0, 1] range
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    # Save output video
    output_filename = "output_vace_inpainting.mp4"
    if not USE_INPAINTING:
        output_filename = "output_vace_r2v.mp4"
    elif not USE_REF_IMAGES:
        output_filename = "output_inpainting.mp4"

    output_path = output_dir / output_filename
    export_to_video(output_video_np, output_path, fps=16)
    print(f"Saved video to {output_path}")

    print("\n=== Generation Statistics ===")
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

    # Save additional visualizations (if inpainting enabled)
    if USE_INPAINTING:
        print("\n=== Saving Additional Visualizations ===")

        # Save mask visualization
        mask_viz_rgb = np.stack([mask, mask, mask], axis=-1)
        export_to_video(
            mask_viz_rgb,
            output_dir / "mask_visualization.mp4",
            fps=16,
        )
        print(f"Saved mask visualization to {output_dir / 'mask_visualization.mp4'}")

        # Save original video for comparison
        original_resized = torch.from_numpy(input_frames).float()
        original_resized = original_resized.permute(0, 3, 1, 2).unsqueeze(0)
        original_resized = torch.nn.functional.interpolate(
            original_resized.squeeze(0),
            size=(config.height, config.width),
            mode="bilinear",
            align_corners=False,
        )
        original_resized = original_resized.permute(0, 2, 3, 1).numpy() / 255.0
        export_to_video(
            original_resized,
            output_dir / "input_original.mp4",
            fps=16,
        )
        print(f"Saved original video to {output_dir / 'input_original.mp4'}")

    print("\n=== Test Complete ===")
    print(f"Results saved to {output_dir}")
    print("\nGenerated files:")
    if USE_INPAINTING:
        print("  - input_original.mp4: Original input video")
        print("  - input_masked_video.mp4: Masked video (gray regions to inpaint)")
        print("  - mask_visualization.mp4: Binary mask (white=inpaint, black=preserve)")
    print(f"  - {output_filename}: Final generated result")
    if USE_REF_IMAGES:
        print(f"  - Used {len(ref_images)} reference image(s) for R2V conditioning")
    if USE_INPAINTING:
        print("  - Used spatial masks for inpainting control")


if __name__ == "__main__":
    main()
