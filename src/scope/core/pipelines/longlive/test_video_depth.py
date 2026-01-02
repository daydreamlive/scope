"""Test script for Video-Depth-Anything preprocessing with LongLive V2V pipeline.

This script demonstrates using Video-Depth-Anything to generate depth maps
from input video, which are then used as VACE depth conditioning for the
LongLive pipeline (V2V workflow).

The flow is: Input Video -> Video-Depth-Anything -> Depth Maps -> LongLive V2V

Usage:
    python -m scope.core.pipelines.longlive.test_video_depth

Configuration:
    Edit the CONFIG dictionary below to set your input video and output paths.
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir
from scope.core.preprocessors import VideoDepthAnything

from ..video import load_video
from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== MODE SELECTION =====
    # "depth_only": Generate NEW video guided by depth structure (T2V + depth)
    # "v2v_depth": Transform input video with depth guidance (V2V + depth)
    "mode": "v2v_depth",  # Options: "depth_only" or "v2v_depth"
    # ===== INPUT =====
    # Path to input video for V2V processing
    "input_video": "frontend/public/assets/test.mp4",
    # ===== VIDEO-DEPTH-ANYTHING SETTINGS =====
    # Encoder size: "vits" (small/fast), "vitb" (base), "vitl" (large/best quality)
    "depth_encoder": "vitl",
    # Number of frames to process per batch (reduce if OOM)
    "depth_batch_size": 32,
    # ===== GENERATION PARAMETERS =====
    # Prompt for video generation (describes the output style/content)
    "prompt": "batman fighting crime",
    "num_chunks": 3,  # Number of generation chunks
    "frames_per_chunk": 12,  # Frames per chunk (12 = 3 latent * 4 temporal upsample)
    "height": 512,
    "width": 512,
    "vace_context_scale": 0.7,  # VACE conditioning strength (0.0-1.0)
    # ===== V2V MODE SETTINGS (only used when mode="v2v_depth") =====
    # Noise scale for V2V (0.0-1.0, higher = more transformation)
    "noise_scale": 0.7,
    "noise_controller": True,
    # ===== OUTPUT =====
    "output_dir": "video_depth_tests",  # path/to/output_dir
    "save_depth_video": True,  # Save the depth map visualization
}

# ========================= END CONFIGURATION =========================


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
    """Load video frames from file.

    Args:
        video_path: Path to video file
        target_height: Target height for resizing
        target_width: Target width for resizing
        max_frames: Maximum number of frames to load (None = all frames)

    Returns:
        Numpy array of shape [F, H, W, C] with values in [0, 255]
    """
    print(f"Loading video: {video_path}")

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

    print(f"Loaded {frames_array.shape[0]} frames at {target_height}x{target_width}")
    return frames_array


def extract_depth_chunk(
    depth_video: torch.Tensor,
    chunk_index: int,
    frames_per_chunk: int,
) -> torch.Tensor:
    """Extract a chunk from depth video tensor.

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


def extract_video_chunk(
    video_frames: list[torch.Tensor],
    chunk_index: int,
    frames_per_chunk: int,
) -> list[torch.Tensor]:
    """Extract a chunk of frames from video.

    Args:
        video_frames: List of video frames [H, W, C]
        chunk_index: Chunk index
        frames_per_chunk: Number of frames per chunk

    Returns:
        List of frames for this chunk, each with shape [1, H, W, C]
    """
    start_idx = chunk_index * frames_per_chunk
    end_idx = start_idx + frames_per_chunk

    # Clamp to video length
    end_idx = min(end_idx, len(video_frames))

    chunk = video_frames[start_idx:end_idx]

    # Pad if needed
    while len(chunk) < frames_per_chunk:
        chunk.append(chunk[-1])

    # Add temporal dimension to each frame: [H, W, C] -> [1, H, W, C]
    chunk = [frame.unsqueeze(0) for frame in chunk]

    return chunk


def main():
    print("=" * 80)
    print("  Video-Depth-Anything + LongLive Test")
    print("=" * 80)

    config = CONFIG
    mode = config.get("mode", "depth_only")
    use_v2v = mode == "v2v_depth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent
    output_dir = resolve_path(config["output_dir"], script_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nConfiguration:")
    print(f"  Mode: {mode}")
    if use_v2v:
        print(f"    → V2V + Depth: Transform input video with depth guidance")
    else:
        print(f"    → Depth-only: Generate new video guided by depth structure")
    print(f"  Input video: {config['input_video']}")
    print(f"  Depth encoder: {config['depth_encoder']}")
    print(f"  Resolution: {config['height']}x{config['width']}")
    print(f"  Chunks: {config['num_chunks']} x {config['frames_per_chunk']} frames")
    print(f"  VACE scale: {config['vace_context_scale']}")
    if use_v2v:
        print(f"  Noise scale: {config['noise_scale']}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}\n")

    # ===== Step 1: Load input video =====
    print("=" * 40)
    print("Step 1: Loading input video")
    print("=" * 40)

    input_video_path = resolve_path(config["input_video"], project_root)
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    total_frames = config["num_chunks"] * config["frames_per_chunk"]
    video_frames_np = load_video_frames(
        input_video_path,
        config["height"],
        config["width"],
        max_frames=total_frames,
    )

    # Convert to list of tensors for pipeline
    video_frames_list = [
        torch.from_numpy(video_frames_np[i]) for i in range(video_frames_np.shape[0])
    ]
    print(f"Prepared {len(video_frames_list)} frames for processing\n")

    # ===== Step 2: Run Video-Depth-Anything =====
    print("=" * 40)
    print("Step 2: Running Video-Depth-Anything")
    print("=" * 40)

    depth_start = time.time()

    # Initialize and load model
    depth_model = VideoDepthAnything(
        encoder=config["depth_encoder"],
        device=device,
        dtype=torch.float16,
    )
    depth_model.load_model()

    # Run depth estimation
    depth_video = depth_model.process_video_for_vace(
        video_frames_np,
        config["height"],
        config["width"],
    )

    depth_time = time.time() - depth_start
    print(f"Depth estimation completed in {depth_time:.2f}s")
    print(f"Depth tensor shape: {depth_video.shape}\n")

    # Save depth visualization if requested
    if config["save_depth_video"]:
        depth_vis = depth_video[0, 0].cpu().numpy()  # [F, H, W]
        depth_vis = ((depth_vis + 1.0) / 2.0 * 255).astype(np.uint8)
        depth_vis_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
        depth_vis_path = output_dir / "depth_visualization.mp4"
        export_to_video(depth_vis_rgb / 255.0, depth_vis_path, fps=16)
        print(f"Saved depth visualization: {depth_vis_path}\n")

    # Offload depth model to free GPU memory for the pipeline
    depth_model.offload()
    del depth_model
    torch.cuda.empty_cache()

    # ===== Step 3: Initialize LongLive pipeline =====
    print("=" * 40)
    print("Step 3: Initializing LongLive pipeline")
    print("=" * 40)

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

    # Set vace_in_dim for depth mode
    pipeline_config.model_config.base_model_kwargs = (
        pipeline_config.model_config.base_model_kwargs or {}
    )
    pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

    pipeline = LongLivePipeline(pipeline_config, device=device, dtype=torch.bfloat16)
    print("Pipeline initialized\n")

    # ===== Step 4: Generate video =====
    print("=" * 40)
    mode_desc = "V2V + Depth" if use_v2v else "Depth-only (T2V)"
    print(f"Step 4: Generating video ({mode_desc})")
    print("=" * 40)

    outputs = []
    latency_measures = []
    fps_measures = []

    frames_per_chunk = config["frames_per_chunk"]

    for chunk_index in range(config["num_chunks"]):
        start_time = time.time()

        # Get depth chunk for this iteration
        depth_chunk = extract_depth_chunk(
            depth_video,
            chunk_index,
            frames_per_chunk,
        )

        # Prepare pipeline kwargs
        kwargs = {
            "prompts": [{"text": config["prompt"], "weight": 100}],
            "vace_input_frames": depth_chunk,  # Depth conditioning from Video-Depth-Anything
            "vace_context_scale": config["vace_context_scale"],
            "init_cache": chunk_index == 0,  # Reset cache on first chunk
        }

        # Add V2V parameters if in v2v_depth mode
        if use_v2v:
            video_chunk = extract_video_chunk(
                video_frames_list,
                chunk_index,
                frames_per_chunk,
            )
            kwargs["video"] = video_chunk
            kwargs["noise_scale"] = config["noise_scale"]
            kwargs["noise_controller"] = config["noise_controller"]

        print(
            f"Chunk {chunk_index}: {'V2V + ' if use_v2v else ''}Depth mode, "
            f"depth shape={depth_chunk.shape}"
        )

        # Generate
        output = pipeline(**kwargs)

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
    output_video = torch.concat(outputs)
    print(f"\nFinal output shape: {output_video.shape}")

    # Save output video
    output_video_np = output_video.contiguous().numpy()
    output_video_np = np.clip(output_video_np, 0.0, 1.0)

    output_filename = f"output_{mode}.mp4"
    output_path = output_dir / output_filename
    export_to_video(output_video_np, output_path, fps=16)
    print(f"\nSaved output: {output_path}")

    # Also save the original input for comparison
    input_save_path = output_dir / "input_video.mp4"
    input_vis = video_frames_np[:output_video.shape[0]].astype(np.float32) / 255.0
    export_to_video(input_vis, input_save_path, fps=16)
    print(f"Saved input: {input_save_path}")

    # ===== Statistics =====
    print("\n" + "=" * 40)
    print("Performance Statistics")
    print("=" * 40)
    print(
        f"Depth estimation: {depth_time:.2f}s "
        f"({total_frames / depth_time:.2f} fps)"
    )
    print(
        f"Generation - Avg latency: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Avg FPS: {sum(fps_measures) / len(fps_measures):.2f}"
    )
    print(f"Total depth + generation time: {depth_time + sum(latency_measures):.2f}s")

    print("\n" + "=" * 80)
    print("  Test Complete")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Input video: input_video.mp4")
    print(f"  - Depth maps: depth_visualization.mp4")
    print(f"  - Output: {output_filename}")


if __name__ == "__main__":
    main()
