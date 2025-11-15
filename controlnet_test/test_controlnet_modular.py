"""
Modular ControlNet test script for multiple pipeline types.

This script supports ControlNet conditioning for:
- LongLive pipeline
- StreamDiffusionV2 pipeline
- Krea Realtime Video pipeline

It uses:
- The Wan ControlNet teacher from notes/wan2.1-dilated-controlnet
- Real pretrained ControlNet weights from HuggingFace
- Control frames extracted from a reference video (canny, depth, etc.)

Usage (example):
    uv run -m controlnet_test.test_controlnet_modular --num-chunks 2
    uv run -m controlnet_test.test_controlnet_modular --num-chunks 2 --config controlnet_test/controlnet_depth.yaml
"""

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from lib.models_config import get_model_file_path, get_models_dir
from lib.schema import Quantization
from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline
from pipelines.longlive.pipeline import LongLivePipeline
from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

from .preprocessors import canny_preprocessor, depth_preprocessor


def _load_controlnet_model(
    config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    controlnet_cfg = config.controlnet
    repo_id = controlnet_cfg.repo_id
    filename = controlnet_cfg.filename
    local_subpath = controlnet_cfg.local_subpath

    controlnet_path = get_model_file_path(local_subpath)
    controlnet_path.parent.mkdir(parents=True, exist_ok=True)

    if not controlnet_path.exists():
        from huggingface_hub import hf_hub_download

        print(
            "test_controlnet_modular: downloading ControlNet weights "
            f"from {repo_id}:{filename}..."
        )
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)

        import shutil

        shutil.copy2(downloaded_path, controlnet_path)
        print(
            "test_controlnet_modular: saved ControlNet weights to " f"{controlnet_path}"
        )

    import safetensors.torch

    state_dict = safetensors.torch.load_file(controlnet_path)

    # Debug: Inspect checkpoint keys to derive architecture
    print("\n" + "=" * 80)
    print("DEBUG: Inspecting ControlNet checkpoint keys")
    print("=" * 80)

    # Find all block indices
    block_indices = set()
    controlnet_block_indices = set()
    for key in state_dict.keys():
        if key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))
        elif key.startswith("controlnet_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                controlnet_block_indices.add(int(parts[1]))

    max_block_idx = max(block_indices) if block_indices else -1
    max_controlnet_block_idx = (
        max(controlnet_block_indices) if controlnet_block_indices else -1
    )
    num_layers_from_checkpoint = max_block_idx + 1
    num_controlnet_blocks_from_checkpoint = max_controlnet_block_idx + 1

    print(
        f"_load_controlnet_model: Found {num_layers_from_checkpoint} blocks in checkpoint"
    )
    print(
        f"_load_controlnet_model: Found {num_controlnet_blocks_from_checkpoint} controlnet_blocks in checkpoint"
    )

    # Inspect controlnet_blocks to derive dimensions
    if "controlnet_blocks.0.weight" in state_dict:
        controlnet_block_weight = state_dict["controlnet_blocks.0.weight"]
        out_proj_dim_from_checkpoint = controlnet_block_weight.shape[0]
        inner_dim_from_checkpoint = controlnet_block_weight.shape[1]
        print(
            f"_load_controlnet_model: controlnet_blocks.0.weight shape: {controlnet_block_weight.shape}"
        )
        print(
            f"_load_controlnet_model: Derived inner_dim={inner_dim_from_checkpoint}, out_proj_dim={out_proj_dim_from_checkpoint}"
        )
    else:
        inner_dim_from_checkpoint = None
        out_proj_dim_from_checkpoint = None
        print(
            "_load_controlnet_model: WARNING: controlnet_blocks.0.weight not found in checkpoint"
        )

    # Inspect patch_embedding to verify inner_dim
    if "patch_embedding.weight" in state_dict:
        patch_embedding_weight = state_dict["patch_embedding.weight"]
        patch_embedding_out_channels = patch_embedding_weight.shape[0]
        print(
            f"_load_controlnet_model: patch_embedding.weight shape: {patch_embedding_weight.shape}"
        )
        print(
            f"_load_controlnet_model: patch_embedding output channels: {patch_embedding_out_channels}"
        )
        if inner_dim_from_checkpoint is None:
            inner_dim_from_checkpoint = patch_embedding_out_channels
        elif inner_dim_from_checkpoint != patch_embedding_out_channels:
            print(
                f"_load_controlnet_model: WARNING: inner_dim mismatch! controlnet_block={inner_dim_from_checkpoint}, patch_embedding={patch_embedding_out_channels}"
            )

    # Derive num_attention_heads from inner_dim
    attention_head_dim = 128  # Standard value
    if inner_dim_from_checkpoint is not None:
        num_attention_heads_from_checkpoint = (
            inner_dim_from_checkpoint // attention_head_dim
        )
        print(
            f"_load_controlnet_model: Derived num_attention_heads={num_attention_heads_from_checkpoint} (from inner_dim={inner_dim_from_checkpoint} / attention_head_dim={attention_head_dim})"
        )
    else:
        num_attention_heads_from_checkpoint = None

    # Inspect ffn_dim from blocks
    if "blocks.0.ffn.net.0.proj.weight" in state_dict:
        ffn_proj_weight = state_dict["blocks.0.ffn.net.0.proj.weight"]
        ffn_dim_from_checkpoint = ffn_proj_weight.shape[0]
        print(
            f"_load_controlnet_model: blocks.0.ffn.net.0.proj.weight shape: {ffn_proj_weight.shape}"
        )
        print(f"_load_controlnet_model: Derived ffn_dim={ffn_dim_from_checkpoint}")
    else:
        ffn_dim_from_checkpoint = None
        print(
            "_load_controlnet_model: WARNING: blocks.0.ffn.net.0.proj.weight not found in checkpoint"
        )

    # Sample some keys to understand structure
    sample_keys = sorted(list(state_dict.keys()))[:20]
    print("_load_controlnet_model: Sample checkpoint keys (first 20):")
    for key in sample_keys:
        shape = state_dict[key].shape if hasattr(state_dict[key], "shape") else "N/A"
        print(f"  {key}: {shape}")

    print("=" * 80 + "\n")

    project_root = Path(__file__).resolve().parents[1]
    controlnet_src = project_root / "notes" / "wan2.1-dilated-controlnet"
    if str(controlnet_src) not in sys.path:
        sys.path.insert(0, str(controlnet_src))

    WanControlnet = import_module("wan_controlnet").WanControlnet  # type: ignore[attr-defined]

    # Use derived values from checkpoint, fallback to defaults
    controlnet_config: dict[str, Any] = {
        "added_kv_proj_dim": None,
        "attention_head_dim": attention_head_dim,
        "cross_attn_norm": True,
        "eps": 1e-06,
        "ffn_dim": ffn_dim_from_checkpoint
        if ffn_dim_from_checkpoint is not None
        else 8960,
        "freq_dim": 256,
        "image_dim": None,
        "in_channels": 3,
        "num_attention_heads": num_attention_heads_from_checkpoint
        if num_attention_heads_from_checkpoint is not None
        else 12,
        "num_layers": num_layers_from_checkpoint
        if num_layers_from_checkpoint > 0
        else 8,
        "patch_size": (1, 2, 2),
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
        "downscale_coef": 8,
        "out_proj_dim": out_proj_dim_from_checkpoint
        if out_proj_dim_from_checkpoint is not None
        else (12 * 128),
    }

    print("_load_controlnet_model: Using ControlNet config:")
    for key, value in controlnet_config.items():
        print(f"  {key}: {value}")
    print()

    controlnet = WanControlnet(**controlnet_config)  # type: ignore[call-arg]

    # Debug: Compare model state dict keys with checkpoint keys
    print("_load_controlnet_model: Comparing model vs checkpoint keys...")
    model_state_dict = controlnet.state_dict()
    model_keys = set(model_state_dict.keys())
    checkpoint_keys = set(state_dict.keys())

    missing_in_checkpoint = model_keys - checkpoint_keys
    missing_in_model = checkpoint_keys - model_keys

    if missing_in_checkpoint:
        print(
            f"_load_controlnet_model: WARNING: {len(missing_in_checkpoint)} keys in model but not in checkpoint:"
        )
        for key in sorted(list(missing_in_checkpoint))[:10]:
            print(f"  {key}")
        if len(missing_in_checkpoint) > 10:
            print(f"  ... and {len(missing_in_checkpoint) - 10} more")

    if missing_in_model:
        print(
            f"_load_controlnet_model: WARNING: {len(missing_in_model)} keys in checkpoint but not in model:"
        )
        for key in sorted(list(missing_in_model))[:10]:
            print(f"  {key}")
        if len(missing_in_model) > 10:
            print(f"  ... and {len(missing_in_model) - 10} more")

    print()

    # Try loading with strict=False first to see what matches
    try:
        missing_keys, unexpected_keys = controlnet.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            print(
                f"_load_controlnet_model: Missing keys after load (strict=False): {len(missing_keys)}"
            )
            for key in sorted(missing_keys)[:20]:
                print(f"  {key}")
            if len(missing_keys) > 20:
                print(f"  ... and {len(missing_keys) - 20} more")
        if unexpected_keys:
            print(
                f"_load_controlnet_model: Unexpected keys after load (strict=False): {len(unexpected_keys)}"
            )
            for key in sorted(unexpected_keys)[:20]:
                print(f"  {key}")
            if len(unexpected_keys) > 20:
                print(f"  ... and {len(unexpected_keys) - 20} more")

        # Now try strict=True to get the actual error
        print(
            "\n_load_controlnet_model: Attempting strict load to see detailed errors..."
        )
        controlnet.load_state_dict(state_dict, strict=True)
        print("_load_controlnet_model: Strict load succeeded!")
    except RuntimeError as e:
        print(f"_load_controlnet_model: ERROR during load_state_dict: {e}")
        print("\n_load_controlnet_model: Attempting to identify specific mismatches...")

        # Try to load each key individually to identify problems
        model_state_dict = controlnet.state_dict()
        shape_mismatches = []
        for key in checkpoint_keys:
            if key in model_state_dict:
                checkpoint_shape = state_dict[key].shape
                model_shape = model_state_dict[key].shape
                if checkpoint_shape != model_shape:
                    shape_mismatches.append((key, checkpoint_shape, model_shape))

        if shape_mismatches:
            print(
                f"_load_controlnet_model: Found {len(shape_mismatches)} shape mismatches:"
            )
            for key, ckpt_shape, model_shape in shape_mismatches[:20]:
                print(f"  {key}: checkpoint={ckpt_shape}, model={model_shape}")
            if len(shape_mismatches) > 20:
                print(f"  ... and {len(shape_mismatches) - 20} more")

        # Re-raise to see full error
        raise

    controlnet = controlnet.to(device=device, dtype=dtype)
    controlnet.eval()

    return controlnet


def _prepare_control_frames(
    config: Any,
    num_teacher_frames: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    preproc_cfg = config.preprocessor
    preproc_type = preproc_cfg.type
    output_video_path = preproc_cfg.output_video_path
    input_video = config.input_video

    if preproc_type == "canny":
        frames_np = canny_preprocessor(
            input_video,
            num_teacher_frames,
            height,
            width,
            output_video_path,
        )
    elif preproc_type == "depth":
        frames_np = depth_preprocessor(
            input_video,
            num_teacher_frames,
            height,
            width,
            output_video_path,
        )
    else:
        raise ValueError(f"Unknown preprocessor type: {preproc_type}")

    if frames_np is None:
        frames_np = np.ones((num_teacher_frames, height, width, 3), dtype=np.float32)

    frames_normalized = frames_np / 255.0
    frames_torch = (
        torch.from_numpy(frames_normalized)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device)
        .to(dtype)
    )
    return frames_torch


def _load_pipeline(
    pipeline_type: str,
    pipeline_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Any:
    """Load pipeline based on type."""
    if pipeline_type == "longlive":
        return LongLivePipeline(
            pipeline_config,
            device=device,
            dtype=dtype,
        )
    elif pipeline_type == "streamdiffusionv2":
        return StreamDiffusionV2Pipeline(
            pipeline_config,
            device=device,
            dtype=dtype,
        )
    elif pipeline_type == "krea_realtime_video":
        quantization = getattr(pipeline_config, "quantization", None)
        if quantization is not None:
            quantization = Quantization[quantization]
        compile_model = getattr(pipeline_config, "compile", False)
        return KreaRealtimeVideoPipeline(
            pipeline_config,
            quantization=quantization,
            compile=compile_model,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def _create_pipeline_config(
    pipeline_type: str,
    experiment_config: Any,
    height: int,
    width: int,
) -> Any:
    """Create pipeline-specific config from experiment config."""
    pipeline_cfg = getattr(experiment_config, "pipeline", {})
    model_dir = str(get_models_dir())

    if pipeline_type == "longlive":
        config_path = Path("pipelines/longlive/model.yaml")
        return OmegaConf.create(
            {
                "model_dir": model_dir,
                "generator_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "generator_path",
                            "LongLive-1.3B/models/longlive_base.pt",
                        )
                    )
                ),
                "lora_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "lora_path",
                            "LongLive-1.3B/models/lora.pt",
                        )
                    )
                ),
                "text_encoder_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "text_encoder_path",
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors",
                        )
                    )
                ),
                "tokenizer_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "tokenizer_path",
                            "Wan2.1-T2V-1.3B/google/umt5-xxl",
                        )
                    )
                ),
                "model_config": OmegaConf.load(str(config_path)),
                "height": height,
                "width": width,
            }
        )
    elif pipeline_type == "streamdiffusionv2":
        config_path = Path("pipelines/streamdiffusionv2/model.yaml")
        return OmegaConf.create(
            {
                "model_dir": model_dir,
                "generator_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "generator_path",
                            "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt",
                        )
                    )
                ),
                "text_encoder_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "text_encoder_path",
                            "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors",
                        )
                    )
                ),
                "tokenizer_path": str(
                    get_model_file_path(
                        getattr(
                            pipeline_cfg,
                            "tokenizer_path",
                            "Wan2.1-T2V-1.3B/google/umt5-xxl",
                        )
                    )
                ),
                "model_config": OmegaConf.load(str(config_path)),
                "height": height,
                "width": width,
            }
        )
    elif pipeline_type == "krea_realtime_video":
        config_path = Path("pipelines/krea_realtime_video/model.yaml")
        config_dict = {
            "model_dir": model_dir,
            "generator_path": str(
                get_model_file_path(
                    getattr(
                        pipeline_cfg,
                        "generator_path",
                        "krea-realtime-video/krea-realtime-video-14b.safetensors",
                    )
                )
            ),
            "text_encoder_path": str(
                get_model_file_path(
                    getattr(
                        pipeline_cfg,
                        "text_encoder_path",
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors",
                    )
                )
            ),
            "tokenizer_path": str(
                get_model_file_path(
                    getattr(
                        pipeline_cfg,
                        "tokenizer_path",
                        "Wan2.1-T2V-1.3B/google/umt5-xxl",
                    )
                )
            ),
            "vae_path": str(
                get_model_file_path(
                    getattr(
                        pipeline_cfg,
                        "vae_path",
                        "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                    )
                )
            ),
            "model_config": OmegaConf.load(str(config_path)),
            "height": height,
            "width": width,
        }
        if hasattr(pipeline_cfg, "quantization"):
            config_dict["quantization"] = pipeline_cfg.quantization
        if hasattr(pipeline_cfg, "compile"):
            config_dict["compile"] = pipeline_cfg.compile
        return OmegaConf.create(config_dict)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Test modular ControlNet conditioning for multiple pipeline types"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="Number of chunks to generate (default: 2)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="controlnet_test/controlnet_canny.yaml",
        help="Path to YAML config for ControlNet experiment",
    )
    args = parser.parse_args()

    experiment_config_path = Path(args.config)
    experiment_config = OmegaConf.load(str(experiment_config_path))
    preprocessor_type = experiment_config.preprocessor.type
    preprocessor_name = preprocessor_type.upper()
    pipeline_type = getattr(experiment_config, "pipeline_type", "longlive")

    print("=" * 80)
    print(f"MODULAR {pipeline_type.upper()} + CONTROLNET {preprocessor_name} TEST")
    print("=" * 80)

    device = torch.device("cuda")
    height = getattr(experiment_config, "height", 480)
    width = getattr(experiment_config, "width", 832)
    num_chunks = args.num_chunks

    # Create pipeline config
    pipeline_config = _create_pipeline_config(
        pipeline_type, experiment_config, height, width
    )

    print(f"\nLoading {pipeline_type} pipeline...")
    pipeline = _load_pipeline(
        pipeline_type,
        pipeline_config,
        device=device,
        dtype=torch.bfloat16,
    )

    controlnet = _load_controlnet_model(
        experiment_config,
        device=device,
        dtype=torch.bfloat16,
    )

    # Prepare ControlNet control frames buffer
    print("\nPreparing ControlNet control frames buffer...")

    # Determine frames per chunk based on pipeline type
    if pipeline_type == "streamdiffusionv2":
        chunk_size = getattr(experiment_config, "chunk_size", 4)
        frames_per_chunk = chunk_size
    elif pipeline_type == "krea_realtime_video":
        frames_per_chunk = getattr(experiment_config, "frames_per_chunk", 3)
    else:  # longlive
        frames_per_chunk = 3

    total_student_frames = num_chunks * frames_per_chunk
    compression_ratio = getattr(experiment_config, "compression_ratio", 4)
    total_teacher_frames = total_student_frames * compression_ratio

    control_frames_buffer = _prepare_control_frames(
        experiment_config,
        total_teacher_frames,
        height,
        width,
        device,
        torch.bfloat16,
    )

    pipeline.state.set("control_frames_buffer", control_frames_buffer)
    pipeline.state.set("controlnet", controlnet)
    conditioning_scale = getattr(experiment_config, "conditioning_scale", 1.0)
    pipeline.state.set("controlnet_weight", conditioning_scale)
    pipeline.state.set("controlnet_stride", 3)
    pipeline.state.set("controlnet_compression_ratio", compression_ratio)

    prompt = getattr(
        experiment_config, "prompt", "A cat sitting in the grass looking back and forth"
    )

    # Handle different pipeline call signatures
    if pipeline_type == "streamdiffusionv2":
        # StreamDiffusionV2 requires video input
        input_video_path = getattr(experiment_config, "input_video", None)
        if input_video_path is None:
            raise ValueError(
                "StreamDiffusionV2 pipeline requires input_video in config"
            )
        chunk_size = getattr(experiment_config, "chunk_size", 4)
        start_chunk_size = getattr(experiment_config, "start_chunk_size", 5)

        # Try torchcodec first, fallback to OpenCV if it fails
        try:
            from pipelines.video import load_video

            input_video = (
                load_video(input_video_path, resize_hw=(height, width))
                .unsqueeze(0)
                .to(device)
                .to(torch.bfloat16)
            )
        except (RuntimeError, ImportError) as e:
            print(
                f"WARNING: torchcodec failed ({e}), using OpenCV fallback for video loading"
            )
            import cv2

            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video file: {input_video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (width, height))
                frames.append(frame_rgb)
            cap.release()

            if not frames:
                raise RuntimeError(
                    f"No frames extracted from video: {input_video_path}"
                )

            # Convert to tensor: [T, H, W, C] -> [C, T, H, W]
            frames_array = np.stack(frames, axis=0)
            frames_tensor = torch.from_numpy(frames_array).float()
            frames_tensor = frames_tensor.permute(
                3, 0, 1, 2
            )  # [T, H, W, C] -> [C, T, H, W]

            # Normalize to [-1, 1]
            frames_tensor = frames_tensor / 127.5 - 1.0

            input_video = frames_tensor.unsqueeze(0).to(device).to(torch.bfloat16)
        _, _, num_frames, _, _ = input_video.shape
        actual_num_chunks = (num_frames - 1) // chunk_size
        if num_chunks > actual_num_chunks:
            print(
                f"WARNING: Requested {num_chunks} chunks but only {actual_num_chunks} available. Using {actual_num_chunks}."
            )
            num_chunks = actual_num_chunks

        # Get optional denoising steps from config
        denoising_step_list = getattr(experiment_config, "denoising_step_list", None)
        if denoising_step_list is not None:
            denoising_step_list = torch.tensor(denoising_step_list, dtype=torch.long)

        print("\nPreparing text condition and caches...")
        start_idx = 0
        end_idx = start_chunk_size
        chunk = input_video[:, :, start_idx:end_idx]
        call_kwargs = {"video": chunk, "prompts": prompt}
        if denoising_step_list is not None:
            call_kwargs["denoising_step_list"] = denoising_step_list
        pipeline(**call_kwargs)

        print(
            f"\nGenerating {num_chunks} chunks with ControlNet {preprocessor_name} conditioning..."
        )
        outputs: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            if chunk_idx > 0:
                start_idx = end_idx
                end_idx = end_idx + chunk_size
            chunk = input_video[:, :, start_idx:end_idx]

            print(f"\nGenerating chunk {chunk_idx + 1}/{num_chunks}...")
            start = time.time()
            call_kwargs = {"video": chunk, "prompts": prompt}
            if denoising_step_list is not None:
                call_kwargs["denoising_step_list"] = denoising_step_list
            output = pipeline(**call_kwargs)
            elapsed = time.time() - start
            print(
                f"Chunk {chunk_idx + 1} generated {output.shape[0]} frames in {elapsed:.2f}s"
            )
            outputs.append(output.detach().cpu())
    else:
        # LongLive and Krea use prompts only
        print("\nPreparing text condition and caches...")
        pipeline(prompts=prompt)

        print(
            f"\nGenerating {num_chunks} chunks with ControlNet {preprocessor_name} conditioning..."
        )
        outputs: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            print(f"\nGenerating chunk {chunk_idx + 1}/{num_chunks}...")
            start = time.time()
            output = pipeline(prompts=prompt)
            elapsed = time.time() - start
            print(
                f"Chunk {chunk_idx + 1} generated {output.shape[0]} frames in {elapsed:.2f}s"
            )
            outputs.append(output.detach().cpu())

    if not outputs:
        print("\nNo output generated - all chunks failed")
        return

    full_output = torch.cat(outputs, dim=0)
    print(f"\nTotal generated frames: {full_output.shape[0]}")

    output_np = full_output.numpy()
    conditioning_scale = getattr(experiment_config, "conditioning_scale", 1.0)
    output_path = f"controlnet_test/output_controlnet_{pipeline_type}_{preprocessor_type}_strength{conditioning_scale:.2f}.mp4"
    export_to_video(output_np, output_path, fps=16)
    print(f"\nSaved video to {output_path}")
    control_frames_path = experiment_config.preprocessor.output_video_path
    print(f"Compare visually against {control_frames_path}")


if __name__ == "__main__":
    main()
