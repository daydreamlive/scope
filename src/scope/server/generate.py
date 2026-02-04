"""Video generation service for batch mode with chunked processing."""

import base64
import gc
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

# Defaults
DEFAULT_HEIGHT = 320
DEFAULT_WIDTH = 576
DEFAULT_CHUNK_SIZE = 12
DEFAULT_SEED = 42
DEFAULT_NOISE_SCALE = 0.7
PROMPT_WEIGHT = 100

if TYPE_CHECKING:
    from logging import Logger

    from .pipeline_manager import PipelineManager
    from .schema import EncodedArray, GenerateRequest


def decode_array(encoded: "EncodedArray", dtype: np.dtype) -> np.ndarray:
    """Decode EncodedArray to numpy array."""
    data = base64.b64decode(encoded.base64)
    return np.frombuffer(data, dtype=dtype).reshape(encoded.shape)


def loop_to_length(arr: np.ndarray, target: int, axis: int) -> np.ndarray:
    """Tile array along axis to reach target length."""
    current = arr.shape[axis]
    if current >= target:
        return arr
    repeats = (target + current - 1) // current
    tiled = np.concatenate([arr] * repeats, axis=axis)
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, target)
    return tiled[tuple(slices)]


def pad_chunk(arr: np.ndarray, target_size: int, axis: int) -> np.ndarray:
    """Pad array with last frame along axis to reach target size."""
    current = arr.shape[axis]
    if current >= target_size:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(-1, None)
    last_frame = arr[tuple(slices)]
    padding = np.repeat(last_frame, target_size - current, axis=axis)
    return np.concatenate([arr, padding], axis=axis)


def build_lookup(specs: list | None, value_attr: str = "image") -> dict:
    """Build chunk -> value lookup from list of specs."""
    if not specs:
        return {}
    return {spec.chunk: getattr(spec, value_attr) for spec in specs}


def get_chunk_value(value, chunk_idx: int, default=None):
    """Get per-chunk value from scalar or list."""
    if value is None:
        return default
    if isinstance(value, list):
        return value[chunk_idx] if chunk_idx < len(value) else value[-1]
    return value


def sse_event(event_type: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@dataclass
class DecodedInputs:
    """Decoded and preprocessed inputs for generation."""

    input_video: np.ndarray | None = None
    vace_frames: np.ndarray | None = None
    vace_masks: np.ndarray | None = None
    first_frames: dict[int, str] = field(default_factory=dict)
    last_frames: dict[int, str] = field(default_factory=dict)
    ref_images: dict[int, list[str]] = field(default_factory=dict)
    prompts: dict[int, str] = field(default_factory=dict)


def decode_inputs(request: "GenerateRequest", num_frames: int) -> DecodedInputs:
    """Decode all base64 inputs from request."""
    inputs = DecodedInputs()

    if request.input_video:
        inputs.input_video = decode_array(request.input_video, np.uint8)
        inputs.input_video = loop_to_length(inputs.input_video, num_frames, axis=0)

    if request.vace_frames:
        inputs.vace_frames = decode_array(request.vace_frames, np.float32)
        inputs.vace_frames = loop_to_length(inputs.vace_frames, num_frames, axis=2)

    if request.vace_masks:
        inputs.vace_masks = decode_array(request.vace_masks, np.float32)
        inputs.vace_masks = loop_to_length(inputs.vace_masks, num_frames, axis=2)

    inputs.first_frames = build_lookup(request.first_frames, "image")
    inputs.last_frames = build_lookup(request.last_frames, "image")
    inputs.ref_images = build_lookup(request.vace_ref_images, "images")
    inputs.prompts = {0: request.prompt}
    inputs.prompts.update(build_lookup(request.chunk_prompts, "text"))

    return inputs


def build_chunk_kwargs(
    request: "GenerateRequest",
    inputs: DecodedInputs,
    chunk_idx: int,
    chunk_size: int,
    start_frame: int,
    end_frame: int,
    status_info: dict,
    device: torch.device,
    dtype: torch.dtype,
    logger: "Logger",
) -> dict:
    """Build pipeline kwargs for a single chunk."""
    kwargs = {
        "height": request.height
        or status_info.get("load_params", {}).get("height", DEFAULT_HEIGHT),
        "width": request.width
        or status_info.get("load_params", {}).get("width", DEFAULT_WIDTH),
        "base_seed": get_chunk_value(request.seed, chunk_idx, DEFAULT_SEED),
        "init_cache": chunk_idx == 0,
        "manage_cache": request.manage_cache,
    }

    # Prompt (sticky behavior - only send when it changes)
    if chunk_idx in inputs.prompts:
        kwargs["prompts"] = [
            {"text": inputs.prompts[chunk_idx], "weight": PROMPT_WEIGHT}
        ]

    if request.denoising_steps:
        kwargs["denoising_step_list"] = request.denoising_steps

    # Video-to-video
    if inputs.input_video is not None:
        chunk_frames = inputs.input_video[start_frame:end_frame]
        chunk_frames = pad_chunk(chunk_frames, chunk_size, axis=0)
        kwargs["video"] = [torch.from_numpy(f).unsqueeze(0) for f in chunk_frames]
        kwargs["noise_scale"] = get_chunk_value(
            request.noise_scale, chunk_idx, DEFAULT_NOISE_SCALE
        )
    else:
        kwargs["num_frames"] = chunk_size

    # VACE context scale
    kwargs["vace_context_scale"] = get_chunk_value(
        request.vace_context_scale, chunk_idx, 1.0
    )

    # LoRA scales
    if request.lora_scales:
        lora_scale_updates = []
        for path, scale_value in request.lora_scales.items():
            scale = get_chunk_value(scale_value, chunk_idx, 1.0)
            lora_scale_updates.append({"path": path, "scale": scale})
            logger.info(
                f"Chunk {chunk_idx}: LoRA scale={scale:.3f} for {Path(path).name}"
            )
        if lora_scale_updates:
            kwargs["lora_scales"] = lora_scale_updates

    # Keyframes
    if chunk_idx in inputs.first_frames:
        kwargs["first_frame_image"] = inputs.first_frames[chunk_idx]
        kwargs["extension_mode"] = (
            "firstlastframe" if chunk_idx in inputs.last_frames else "firstframe"
        )

    if chunk_idx in inputs.last_frames:
        kwargs["last_frame_image"] = inputs.last_frames[chunk_idx]
        if chunk_idx not in inputs.first_frames:
            kwargs["extension_mode"] = "lastframe"

    if chunk_idx in inputs.ref_images:
        kwargs["vace_ref_images"] = inputs.ref_images[chunk_idx]

    # VACE conditioning frames [1, C, T, H, W]
    if inputs.vace_frames is not None:
        chunk = inputs.vace_frames[:, :, start_frame:end_frame, :, :]
        chunk = pad_chunk(chunk, chunk_size, axis=2)
        kwargs["vace_input_frames"] = torch.from_numpy(chunk).to(device, dtype)

    # VACE masks [1, 1, T, H, W]
    if inputs.vace_masks is not None:
        chunk = inputs.vace_masks[:, :, start_frame:end_frame, :, :]
        chunk = pad_chunk(chunk, chunk_size, axis=2)
        kwargs["vace_input_masks"] = torch.from_numpy(chunk).to(device, dtype)

    return kwargs


def generate_video_stream(
    request: "GenerateRequest",
    pipeline_manager: "PipelineManager",
    status_info: dict,
    logger: "Logger",
) -> Iterator[str]:
    """Generate video frames, yielding SSE events."""
    try:
        pipeline = pipeline_manager.get_pipeline_by_id(request.pipeline_id)

        # Determine chunk size from pipeline
        has_video = request.input_video is not None
        requirements = pipeline.prepare(video=[] if has_video else None)
        chunk_size = requirements.input_size if requirements else DEFAULT_CHUNK_SIZE
        num_chunks = (request.num_frames + chunk_size - 1) // chunk_size

        # Decode inputs
        inputs = decode_inputs(request, request.num_frames)

        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        output_chunks = []
        latency_measures = []
        fps_measures = []

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, request.num_frames)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            kwargs = build_chunk_kwargs(
                request,
                inputs,
                chunk_idx,
                chunk_size,
                start_frame,
                end_frame,
                status_info,
                device,
                dtype,
                logger,
            )

            # Run pipeline
            chunk_start = time.time()
            with torch.amp.autocast("cuda", dtype=dtype):
                result = pipeline(**kwargs)
            chunk_latency = time.time() - chunk_start

            chunk_output = result["video"]
            num_output_frames = chunk_output.shape[0]
            chunk_fps = num_output_frames / chunk_latency

            latency_measures.append(chunk_latency)
            fps_measures.append(chunk_fps)

            logger.info(
                f"Chunk {chunk_idx + 1}/{num_chunks}: "
                f"{num_output_frames} frames, latency={chunk_latency:.2f}s, fps={chunk_fps:.2f}"
            )

            output_chunks.append(chunk_output.detach().cpu())

            yield sse_event(
                "progress",
                {
                    "chunk": chunk_idx + 1,
                    "total_chunks": num_chunks,
                    "frames": num_output_frames,
                    "latency": round(chunk_latency, 3),
                    "fps": round(chunk_fps, 2),
                },
            )

        # Concatenate and encode output
        output_video = torch.cat(output_chunks, dim=0)
        output_np = output_video.numpy()

        # Log performance summary
        if latency_measures:
            avg_latency = sum(latency_measures) / len(latency_measures)
            avg_fps = sum(fps_measures) / len(fps_measures)
            logger.info(
                f"=== Performance Summary ({num_chunks} chunks) ===\n"
                f"  Latency - Avg: {avg_latency:.2f}s, "
                f"Max: {max(latency_measures):.2f}s, Min: {min(latency_measures):.2f}s\n"
                f"  FPS - Avg: {avg_fps:.2f}, "
                f"Max: {max(fps_measures):.2f}, Min: {min(fps_measures):.2f}"
            )

        video_bytes = output_np.astype(np.float32).tobytes()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")

        yield sse_event(
            "complete",
            {
                "video_base64": video_base64,
                "video_shape": list(output_np.shape),
                "num_frames": output_np.shape[0],
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
            },
        )

    except Exception as e:
        logger.exception("Error generating video")
        yield sse_event("error", {"error": str(e)})
