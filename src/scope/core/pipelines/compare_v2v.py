import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.transforms import v2

from scope.core.config import get_model_file_path, get_models_dir

# Configure logging - set root to WARNING to keep non-app libraries quiet by default
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler handles INFO
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)

# Set INFO level for app modules
logging.getLogger("scope.core").setLevel(logging.INFO)
logging.getLogger("scope.server").setLevel(logging.INFO)


def load_video_cv2(
    path: str,
    num_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Loads a video as a CTHW tensor using OpenCV (Windows-compatible).
    """
    cap = cv2.VideoCapture(str(path))
    frames = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
        if num_frames is not None and frame_count >= num_frames:
            break

    cap.release()

    # Stack frames: list of HWC -> THWC
    video = torch.from_numpy(np.stack(frames))
    # Rearrange to TCHW for resize
    video = rearrange(video, "T H W C -> T C H W")

    height, width = video.shape[2:]
    if resize_hw is not None and (height != resize_hw[0] or width != resize_hw[1]):
        video = v2.Resize(resize_hw, antialias=True)(video)

    video = video.float()

    if normalize:
        # Normalize to [-1, 1]
        video = video / 127.5 - 1.0

    # Rearrange to CTHW
    video = rearrange(video, "T C H W -> C T H W")

    return video


def test_pipeline(
    pipeline_name: str,
    max_frames: int = 96,
    num_chunks: int = None,
    input_mode: str = "video",
):
    """Test a pipeline with video-to-video generation.

    Args:
        pipeline_name: Name of the pipeline to test (longlive, reward_forcing, or krea)
        max_frames: Maximum number of frames to load from the video
        num_chunks: Number of chunks to process. If None, automatically calculated based on video length.
        input_mode: Input mode ("text" or "video"). Defaults to "video" for v2v. Note: backend infers mode from video parameter presence.
    """
    print(f"\n{'='*80}")
    print(f"Testing {pipeline_name} pipeline")
    print(f"{'='*80}\n")

    if pipeline_name == "longlive":
        from scope.core.pipelines.longlive.pipeline import LongLivePipeline

        chunk_size = 12
        start_chunk_size = 12

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
                "model_config": OmegaConf.load(
                    Path(__file__).parent / "longlive" / "model.yaml"
                ),
                "height": 512,
                "width": 512,
            }
        )
        pipeline = LongLivePipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )

    elif pipeline_name == "reward_forcing":
        from scope.core.pipelines.reward_forcing.pipeline import RewardForcingPipeline

        chunk_size = 12
        start_chunk_size = 12

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
                "model_config": OmegaConf.load(
                    Path(__file__).parent / "reward_forcing" / "model.yaml"
                ),
                "height": 512,
                "width": 512,
            }
        )
        pipeline = RewardForcingPipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )

    elif pipeline_name == "krea":
        from scope.core.pipelines.krea_realtime_video.pipeline import (
            KreaRealtimeVideoPipeline,
        )
        from scope.core.pipelines.utils import Quantization

        chunk_size = 12
        start_chunk_size = 12

        config = OmegaConf.create(
            {
                "model_dir": str(get_models_dir()),
                "generator_path": str(
                    get_model_file_path(
                        "krea-realtime-video/krea-realtime-video-14b.safetensors"
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
                "vae_path": str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")),
                "model_config": OmegaConf.load(
                    Path(__file__).parent / "krea_realtime_video" / "model.yaml"
                ),
                "height": 256,
                "width": 256,
            }
        )
        pipeline = KreaRealtimeVideoPipeline(
            config,
            quantization=Quantization.FP8_E4M3FN,
            compile=False,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    # Load the default test video from the frontend assets
    default_video_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "frontend"
        / "public"
        / "assets"
        / "test.mp4"
    )
    print(f"compare_v2v: Loading video from {default_video_path}")

    input_video = (
        load_video_cv2(
            default_video_path,
            num_frames=max_frames,
            resize_hw=(config.height, config.width),
        )
        .unsqueeze(0)
        .to("cuda", torch.bfloat16)
    )
    _, _, num_frames, _, _ = input_video.shape
    print(
        f"compare_v2v: Loaded video with {num_frames} frames, shape={input_video.shape}"
    )

    if num_chunks is None:
        num_chunks = (num_frames - start_chunk_size) // chunk_size + 1
        print(
            f"compare_v2v: Auto-calculated {num_chunks} chunks (chunk_size={chunk_size})"
        )
    else:
        print(
            f"compare_v2v: Processing {num_chunks} chunks as specified (chunk_size={chunk_size})"
        )

    prompts = [
        {
            "text": "A 3D animated scene. A **panda** sitting in the grass, looking around.",
            "weight": 100,
        }
    ]

    outputs = []
    latency_measures = []
    fps_measures = []
    start_idx = 0
    end_idx = start_chunk_size

    for i in range(num_chunks):
        if i > 0:
            start_idx = end_idx
            end_idx = end_idx + chunk_size

        # Calculate requested chunk size
        requested_chunk_size = chunk_size if i > 0 else start_chunk_size

        # Extract chunk with looping support
        chunk_parts = []
        remaining_frames = requested_chunk_size
        current_start = start_idx

        while remaining_frames > 0:
            # Wrap start index if it exceeds video length
            current_start_wrapped = current_start % num_frames
            # Calculate how many frames we can take from current position
            frames_available = num_frames - current_start_wrapped
            frames_to_take = min(remaining_frames, frames_available)

            # Extract frames from current position
            part = input_video[
                :, :, current_start_wrapped : current_start_wrapped + frames_to_take
            ]
            chunk_parts.append(part)

            remaining_frames -= frames_to_take
            current_start += frames_to_take

        # Concatenate parts if chunk wrapped around
        if len(chunk_parts) > 1:
            chunk = torch.cat(chunk_parts, dim=2)
        else:
            chunk = chunk_parts[0]

        chunk_frames = chunk.shape[2]

        print(
            f"compare_v2v: Processing chunk {i + 1}/{num_chunks} (start_idx={start_idx}, requested_size={requested_chunk_size}, actual_size={chunk_frames})"
        )

        start = time.time()
        # output is THWC
        # Pass input_mode for consistency with frontend (backend infers from video parameter)
        if pipeline_name == "krea":
            output = pipeline(
                video=chunk,
                prompts=prompts,
                kv_cache_attention_bias=0.01,
                denoising_step_list=[1000, 750, 500, 250],
                noise_scale=0.7,
                noise_controller=True,
            )
        else:
            output = pipeline(
                video=chunk,
                prompts=prompts,
                noise_scale=0.7,
                noise_controller=True,
                input_mode=input_mode,
            )

        num_output_frames, _, _, _ = output.shape
        latency = time.time() - start
        fps = num_output_frames / latency

        print(
            f"compare_v2v: Pipeline generated {num_output_frames} frames latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        outputs.append(output.detach().cpu())

    # Concatenate all of the THWC tensors
    output_video = torch.concat(outputs)
    print(f"compare_v2v: Final output shape={output_video.shape}")
    output_video_np = output_video.contiguous().numpy()
    output_path = Path(__file__).parent / f"compare_output_v2v_{pipeline_name}.mp4"
    export_to_video(output_video_np, output_path, fps=16)

    # Print statistics
    print("\n=== Performance Statistics ===")
    print(f"Total chunks processed: {len(latency_measures)}")
    print(f"Total frames processed: {sum(o.shape[0] for o in outputs)}")
    print(
        f"Latency - Avg: {sum(latency_measures) / len(latency_measures):.2f}s, "
        f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s"
    )
    print(
        f"FPS - Avg: {sum(fps_measures) / len(fps_measures):.2f}, "
        f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
    )
    print(f"\nOutput saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test video-to-video pipelines with chunked processing"
    )
    parser.add_argument(
        "pipeline_name",
        nargs="+",
        choices=["longlive", "reward_forcing", "krea"],
        help="Name(s) of the pipeline(s) to test. Can specify multiple pipelines.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=96,
        help="Maximum number of frames to load from video (default: 96)",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=None,
        help="Number of chunks to process (default: auto-calculated based on video length)",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        choices=["text", "video"],
        default="video",
        help='Input mode: "text" for text-to-video or "video" for video-to-video (default: video). Note: backend infers mode from video parameter presence.',
    )

    args = parser.parse_args()

    # Process each pipeline
    for pipeline_name in args.pipeline_name:
        test_pipeline(pipeline_name, args.max_frames, args.num_chunks, args.input_mode)
