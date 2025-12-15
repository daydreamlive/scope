"""
Test script for LongLive inpainting (masked video-to-video generation).

This script demonstrates:
1. Loading an input video and a mask video
2. Creating masked video frames (gray out regions to inpaint)
3. Generating video with inpainting using LongLive pipeline via standard VACE path
4. Comparing inpainted vs original video

Follows original VACE architecture (notes/VACE/vace/models/wan/wan_vace.py):
- input_frames = original video with masked regions grayed out (3-channel RGB)
- input_masks = spatial control masks (white=generate, black=preserve)
- Standard encoding: vace_encode_frames (with masks) -> vace_encode_masks -> vace_latent

Requirements:
- Input video: frontend/public/assets/test.mp4
- Mask video: src/scope/core/pipelines/longlive/test_depth_video.mp4

Usage:
    python -m scope.core.pipelines.longlive.test_inpainting
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


def extract_chunk(
    video_tensor: torch.Tensor, chunk_index: int, frames_per_chunk: int
) -> torch.Tensor:
    """
    Extract a chunk from video tensor.

    Args:
        video_tensor: Video tensor [1, C, F, H, W]
        chunk_index: Chunk index
        frames_per_chunk: Number of frames per chunk

    Returns:
        Chunk tensor [1, C, frames_per_chunk, H, W]
    """
    start_idx = chunk_index * frames_per_chunk
    end_idx = start_idx + frames_per_chunk

    # Clamp to video length
    total_frames = video_tensor.shape[2]
    end_idx = min(end_idx, total_frames)

    chunk = video_tensor[:, :, start_idx:end_idx, :, :]

    # Pad if needed
    if chunk.shape[2] < frames_per_chunk:
        padding = frames_per_chunk - chunk.shape[2]
        pad_frames = chunk[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)
        chunk = torch.cat([chunk, pad_frames], dim=2)

    return chunk


def main():
    print("=== LongLive Inpainting Test ===\n")

    # Paths
    # __file__ is in src/scope/core/pipelines/longlive/
    # Go up 6 levels to get to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    input_video_path = project_root / "frontend" / "public" / "assets" / "test.mp4"
    mask_video_path = (
        project_root
        / "src"
        / "scope"
        / "core"
        / "pipelines"
        / "longlive"
        / "vace_tests"
        # / "white_mask_512x512.mp4"
        / "static_mask_half_white_half_black.mp4"
        # / "white_square_moving.mp4"
    )
    output_dir = (
        project_root
        / "src"
        / "scope"
        / "core"
        / "pipelines"
        / "longlive"
        / "vace_tests"
        / "inpainting"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Input video: {input_video_path}")
    print(f"Mask video: {mask_video_path}")
    print(f"Output dir: {output_dir}\n")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "generator_path": str(
                get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
            ),
            # Skip LoRA for this test to avoid rank mismatch issues
            # "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
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
            "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
            "height": 512,
            "width": 512,
        }
    )

    # Override vace_in_dim for inpainting mode
    config.model_config.base_model_kwargs = config.model_config.base_model_kwargs or {}
    config.model_config.base_model_kwargs["vace_in_dim"] = (
        96  # Use 96 to load pretrained VACE weights
    )

    # Load pipeline
    print("Loading pipeline...")
    pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
    print("Pipeline loaded\n")

    # Parameters
    prompt_text = "A fireball, high quality, cinematic"
    num_chunks = 3
    frames_per_chunk = 12
    total_frames = num_chunks * frames_per_chunk

    print(f"Prompt: {prompt_text}")
    print(
        f"Generating {num_chunks} chunks, {frames_per_chunk} frames per chunk = {total_frames} total frames\n"
    )

    # Load input video
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
    # Convert uint8 [0, 255] to float [0, 1] for export_to_video
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

    # Generate with inpainting
    print("=== Generating with Inpainting ===")
    outputs_inpaint = []
    latency_measures = []
    fps_measures = []

    # Use overlapping chunks for smooth transitions (LongLive expects 3-frame overlap)
    # Chunk 0: frames 0-11, Chunk 1: frames 9-20, Chunk 2: frames 18-29, etc.
    overlap_frames = 3  # LongLive uses 3-frame overlap for blending
    start_idx = 0

    for chunk_index in range(num_chunks):
        start = time.time()

        # Calculate chunk boundaries with overlap
        if chunk_index > 0:
            # Start 3 frames before previous chunk ended (overlap)
            start_idx = start_idx + frames_per_chunk - overlap_frames
        end_idx = start_idx + frames_per_chunk

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

        print(
            f"\nChunk {chunk_index}: start_idx={start_idx}, end_idx={end_idx_clamped}, overlap={overlap_frames if chunk_index > 0 else 0}, input_chunk shape = {input_chunk.shape}, mask_chunk shape = {mask_chunk.shape}"
        )

        # Generate with inpainting
        output = pipeline(
            prompts=[{"text": prompt_text, "weight": 100}],
            input_frames=input_chunk,
            input_masks=mask_chunk,
            vace_context_scale=1.0,
        )

        num_output_frames_chunk, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames_chunk / latency

        print(
            f"Chunk {chunk_index}: Generated {num_output_frames_chunk} frames "
            f"latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs_inpaint.append(output.detach().cpu())

    # Save inpainted video
    # Concatenate chunks, skipping overlapping frames from subsequent chunks
    # Chunk 0: take all frames, Chunk 1+: skip first overlap_frames frames
    output_chunks = []
    for chunk_idx, output_chunk in enumerate(outputs_inpaint):
        if chunk_idx == 0:
            # First chunk: take all frames
            output_chunks.append(output_chunk)
        else:
            # Subsequent chunks: skip overlapping frames
            output_chunks.append(output_chunk[overlap_frames:])

    output_video_inpaint = torch.concat(output_chunks)
    print(f"\nInpainted output shape: {output_video_inpaint.shape}")
    output_video_inpaint_np = output_video_inpaint.contiguous().numpy()
    # Ensure values are in [0, 1] range (pipeline outputs should already be normalized)
    output_video_inpaint_np = np.clip(output_video_inpaint_np, 0.0, 1.0)
    export_to_video(
        output_video_inpaint_np,
        output_dir / "output_inpainted.mp4",
        fps=16,
    )

    print("\n=== Inpainting Generation Statistics ===")
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

    # Save mask visualization
    print("\n=== Saving Mask Visualization ===")
    # Keep mask as float [0, 1] and convert to RGB
    mask_viz_rgb = np.stack([mask, mask, mask], axis=-1)
    export_to_video(
        mask_viz_rgb,
        output_dir / "mask_visualization.mp4",
        fps=16,
    )
    print(f"Saved mask visualization to {output_dir / 'mask_visualization.mp4'}")

    # Save original video for comparison
    print("\n=== Saving Original Video for Comparison ===")
    original_resized = torch.from_numpy(input_frames).float()
    original_resized = original_resized.permute(0, 3, 1, 2).unsqueeze(0)
    original_resized = torch.nn.functional.interpolate(
        original_resized.squeeze(0),
        size=(config.height, config.width),
        mode="bilinear",
        align_corners=False,
    )
    # Keep as float [0, 1] for export_to_video (input_frames is [0, 255], so normalize)
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
    print("  - input_original.mp4: Original input video")
    print("  - input_masked_video.mp4: Masked video (gray regions to inpaint)")
    print("  - mask_visualization.mp4: Binary mask (white=inpaint, black=preserve)")
    print("  - output_inpainted.mp4: Final inpainted result")


if __name__ == "__main__":
    main()
