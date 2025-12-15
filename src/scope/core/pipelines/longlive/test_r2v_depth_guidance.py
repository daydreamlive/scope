"""
Test script combining VACE R2V (Reference-to-Video) and depth guidance.

This script demonstrates:
1. Using reference images for style/content conditioning (R2V)
2. Using depth maps for structural guidance (depth guidance)
3. Combining both conditioning methods simultaneously

Requirements:
- VACE model weights at ~/.daydream-scope/models/Wan2.1-VACE-1.3B/
- Depth video file at vace_tests/control_frames_depth.mp4
- Reference image(s) for R2V conditioning
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
from .vace_utils import extract_depth_chunk, preprocess_depth_frames


def load_video_frames(video_path, target_height=480, target_width=832, max_frames=None):
    """Load video frames from file."""
    print(f"load_video_frames: Loading video from {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            frame = cv2.resize(frame, (target_width, target_height))

        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()

    print(
        f"load_video_frames: Loaded {len(frames)} frames at {target_height}x{target_width}"
    )
    return np.array(frames)


def main():
    print("test_r2v_depth_guidance: Starting main() function...")
    try:
        # Configuration
        output_dir = Path(__file__).parent / "vace_tests" / "r2v_depth_guidance"
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"test_r2v_depth_guidance: Output directory: {output_dir}")

        # Check if VACE model is available
        vace_model_path = (
            Path.home() / ".daydream-scope" / "models" / "Wan2.1-VACE-1.3B"
        )
        vace_checkpoint = vace_model_path / "diffusion_pytorch_model.safetensors"

        if not vace_checkpoint.exists():
            print(f"VACE checkpoint not found at {vace_checkpoint}")
            print(
                "Please download VACE weights to ~/.daydream-scope/models/Wan2.1-VACE-1.3B/"
            )
            raise FileNotFoundError(f"VACE checkpoint not found at {vace_checkpoint}")

        vace_path = str(vace_checkpoint)
        print(f"test_r2v_depth_guidance: Using VACE model from {vace_path}")

        # Pipeline configuration
        print("test_r2v_depth_guidance: Loading configuration...")
        config = OmegaConf.create(
            {
                "model_dir": str(get_models_dir()),
                "generator_path": str(
                    get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
                ),
                "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
                "text_encoder_path": str(
                    get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
                "vace_path": vace_path,
                "model_config": OmegaConf.load(Path(__file__).parent / "model.yaml"),
                "height": 480,
                "width": 832,
            }
        )

        # Override vace_in_dim for depth mode
        config.model_config.base_model_kwargs = (
            config.model_config.base_model_kwargs or {}
        )
        config.model_config.base_model_kwargs["vace_in_dim"] = (
            96  # Use 96 to load pretrained R2V weights
        )

        print("test_r2v_depth_guidance: Initializing pipeline...")
        device = torch.device("cuda")
        pipeline = LongLivePipeline(config, device=device, dtype=torch.bfloat16)
        print("test_r2v_depth_guidance: Pipeline initialized successfully")

        # Test parameters
        prompt_text = ""
        num_output_frames = 80
        max_output_frames = 80

        # Reference images for R2V conditioning
        ref_images = [
            "C:/_dev/projects/scope/example.png"
            # Add more reference images as needed
        ]

        # If no reference images provided, try to find example.png in project root
        if not ref_images or not all(Path(p).exists() for p in ref_images):
            example_img = (
                Path(__file__).parent.parent.parent.parent.parent / "example.png"
            )
            if example_img.exists():
                ref_images = [str(example_img)]
                print(f"test_r2v_depth_guidance: Using example image: {example_img}")
            else:
                print(
                    "test_r2v_depth_guidance: Warning: No reference images found. R2V conditioning will be disabled."
                )
                ref_images = []

        # Load depth maps from existing video
        depth_video_path = (
            Path(__file__).parent / "vace_tests" / "control_frames_depth.mp4"
        )
        if not depth_video_path.exists():
            raise FileNotFoundError(f"Depth video not found at {depth_video_path}")

        print(f"test_r2v_depth_guidance: Loading depth maps from {depth_video_path}")
        depth_frames_rgb = load_video_frames(
            depth_video_path,
            target_height=config.height,
            target_width=config.width,
            max_frames=num_output_frames,
        )  # [F, H, W, C]

        # Convert RGB to grayscale (depth maps are single-channel)
        # Take the first channel since depth videos are typically grayscale saved as RGB
        depth_frames_np = depth_frames_rgb[:, :, :, 0]  # [F, H, W]
        # Normalize to [0, 1] range
        depth_frames_np = depth_frames_np.astype(np.float32) / 255.0

        # Preprocess depth frames
        depth_frames_tensor = torch.from_numpy(depth_frames_np).float()  # [F, H, W]
        depth_video = preprocess_depth_frames(
            depth_frames_tensor,
            config.height,
            config.width,
            device,
        )  # [1, 3, F, H, W]

        print(f"test_r2v_depth_guidance: Depth video shape: {depth_video.shape}")

        # Generate video with combined R2V + depth guidance
        print("\n=== Generating video with R2V + depth guidance ===")
        outputs = []
        latency_measures = []
        fps_measures = []

        num_frames_generated = 0
        chunk_index = 0
        frames_per_chunk = 12  # 3 latent frames * 4 temporal upsample
        is_first_chunk = True

        while num_frames_generated < max_output_frames:
            start = time.time()

            # Check if we have enough depth frames for this chunk
            depth_frames_available = depth_video.shape[2]
            depth_frames_needed = chunk_index * frames_per_chunk + frames_per_chunk
            if depth_frames_needed > depth_frames_available:
                print(
                    f"\nStopping: Not enough depth frames for chunk {chunk_index}. "
                    f"Need {depth_frames_needed}, have {depth_frames_available}"
                )
                break

            # Extract depth chunk for this generation step
            depth_chunk = extract_depth_chunk(
                depth_video,
                chunk_index,
                frames_per_chunk,
            )

            print(f"\nChunk {chunk_index}: depth_chunk shape = {depth_chunk.shape}")

            # Prepare kwargs for pipeline
            prompts = [{"text": prompt_text, "weight": 100}]
            kwargs = {
                "prompts": prompts,
                "input_frames": depth_chunk,
                "vace_context_scale": 0.7,
            }

            # Add reference images only on first chunk (R2V conditioning)
            if is_first_chunk and ref_images:
                kwargs["ref_images"] = ref_images  # List of paths
                kwargs["vace_context_scale"] = 1.0  # Higher scale when using ref images
                print(
                    f"Chunk {chunk_index}: Using {len(ref_images)} reference image(s)"
                )

            # Generate with combined conditioning
            output = pipeline(**kwargs)
            is_first_chunk = False

            num_output_frames_chunk, _, _, _ = output.shape
            latency = time.time() - start
            fps = num_output_frames_chunk / latency

            print(
                f"Chunk {chunk_index}: Generated {num_output_frames_chunk} frames "
                f"latency={latency:.2f}s fps={fps:.2f}"
            )

            latency_measures.append(latency)
            fps_measures.append(fps)
            num_frames_generated += num_output_frames_chunk
            chunk_index += 1
            outputs.append(output.detach().cpu())

        # Concatenate all outputs
        output_video = torch.concat(outputs)
        has_nan = torch.isnan(output_video).any().item()
        has_inf = torch.isinf(output_video).any().item()
        print(
            f"\nFinal output: shape={output_video.shape}, nan={has_nan}, inf={has_inf}, "
            f"range=[{output_video.min().item():.2f},{output_video.max().item():.2f}]"
        )

        # Export video
        output_path = output_dir / "output_r2v_depth_guided.mp4"
        output_video_np = output_video.contiguous().numpy()
        export_to_video(output_video_np, output_path, fps=16)
        print(f"\nSaved video to {output_path}")

        # Save depth visualization
        print("\nSaving depth visualization...")
        depth_vis = (depth_frames_np[:num_frames_generated] * 255).astype(np.uint8)
        depth_vis_rgb = np.stack(
            [depth_vis, depth_vis, depth_vis], axis=-1
        )  # Convert to RGB
        export_to_video(
            depth_vis_rgb,
            output_dir / "depth_maps.mp4",
            fps=16,
        )

        # Print statistics
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

        print("\n=== R2V + Depth Guidance Test Complete ===")
        if ref_images:
            print(f"Used {len(ref_images)} reference image(s) for R2V conditioning")
        else:
            print("No reference images used (depth guidance only)")
        print("Used depth maps for structural guidance")
        print(f"\nOutputs saved to {output_dir}/")
        print(
            "- output_r2v_depth_guided.mp4: Generated video with R2V + depth guidance"
        )
        print("- depth_maps.mp4: Depth map visualization")
    except Exception as e:
        print(f"test_r2v_depth_guidance: Error occurred: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    print("test_r2v_depth_guidance: Script entry point reached")
    try:
        main()
        print("test_r2v_depth_guidance: main() completed successfully")
    except SystemExit as e:
        print(f"test_r2v_depth_guidance: SystemExit caught: {e}")
        raise
    except Exception as e:
        print(f"test_r2v_depth_guidance: Unhandled exception in __main__: {e}")
        import traceback

        traceback.print_exc()
        raise
