"""Video generation service for batch mode with chunked processing."""

import base64
import gc
import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

# Cancellation support (single-client, so one event suffices)
_cancel_event = threading.Event()


def cancel_generation():
    """Signal the current generation to stop after the current chunk."""
    _cancel_event.set()


def is_generation_cancelled() -> bool:
    """Check if cancellation has been requested."""
    return _cancel_event.is_set()


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
    prompts: dict[int, list[dict]] = field(default_factory=dict)
    transitions: dict[int, dict] = field(default_factory=dict)
    vace_chunk_specs: dict[int, dict] = field(default_factory=dict)


def load_video_from_file(file_path: str) -> np.ndarray:
    """Load video from temp file.

    Args:
        file_path: Path to video file with header

    Returns:
        Video array [T, H, W, C] uint8
    """
    with open(file_path, "rb") as f:
        ndim = int.from_bytes(f.read(4), "little")
        shape = tuple(int.from_bytes(f.read(4), "little") for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data


def decode_inputs(
    request: "GenerateRequest", num_frames: int, logger: "Logger"
) -> DecodedInputs:
    """Decode all inputs from request (base64 or file-based)."""
    inputs = DecodedInputs()

    # Handle input video - either from file path or base64
    if request.input_path:
        logger.info(f"Loading input video from file: {request.input_path}")
        inputs.input_video = load_video_from_file(request.input_path)
        inputs.input_video = loop_to_length(inputs.input_video, num_frames, axis=0)
    elif request.input_video:
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
    # Normalize prompt to weighted list format
    if isinstance(request.prompt, str):
        inputs.prompts = {0: [{"text": request.prompt, "weight": PROMPT_WEIGHT}]}
    else:
        inputs.prompts = {
            0: [{"text": p.text, "weight": p.weight} for p in request.prompt]
        }

    # Chunk prompts: support both text and weighted prompt lists
    if request.chunk_prompts:
        for spec in request.chunk_prompts:
            if spec.prompts:
                inputs.prompts[spec.chunk] = [
                    {"text": p.text, "weight": p.weight} for p in spec.prompts
                ]
            elif spec.text:
                inputs.prompts[spec.chunk] = [
                    {"text": spec.text, "weight": PROMPT_WEIGHT}
                ]

    # Per-chunk VACE specs
    if request.vace_chunk_specs:
        logger.info(
            f"decode_inputs: Found {len(request.vace_chunk_specs)} vace_chunk_specs"
        )
        for spec in request.vace_chunk_specs:
            logger.info(
                f"decode_inputs: vace_chunk_spec chunk={spec.chunk}, has_frames={spec.frames is not None}, has_masks={spec.masks is not None}, context_scale={spec.context_scale}, temporally_locked={spec.vace_temporally_locked}"
            )
            decoded_spec: dict = {
                "vace_temporally_locked": spec.vace_temporally_locked,
            }
            if spec.frames is not None:
                decoded_spec["frames"] = decode_array(spec.frames, np.float32)
                logger.info(
                    f"decode_inputs: chunk {spec.chunk} decoded frames shape={decoded_spec['frames'].shape}"
                )
            if spec.masks is not None:
                decoded_spec["masks"] = decode_array(spec.masks, np.float32)
                logger.info(
                    f"decode_inputs: chunk {spec.chunk} decoded masks shape={decoded_spec['masks'].shape}"
                )
            if spec.context_scale is not None:
                decoded_spec["context_scale"] = spec.context_scale
            inputs.vace_chunk_specs[spec.chunk] = decoded_spec
        logger.info(
            f"decode_inputs: vace_chunk_specs keys={list(inputs.vace_chunk_specs.keys())}"
        )
    else:
        logger.info("decode_inputs: No vace_chunk_specs in request")

    # Build transitions lookup
    if request.transitions:
        for t in request.transitions:
            inputs.transitions[t.chunk] = {
                "target_prompts": [
                    {"text": p.text, "weight": p.weight} for p in t.target_prompts
                ],
                "num_steps": t.num_steps,
                "temporal_interpolation_method": t.temporal_interpolation_method,
            }

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
        "init_cache": chunk_idx == 0
        or (
            request.cache_reset_chunks is not None
            and chunk_idx in request.cache_reset_chunks
        ),
        "manage_cache": request.manage_cache,
    }

    # Prompt (sticky behavior - only send when it changes)
    if chunk_idx in inputs.prompts:
        kwargs["prompts"] = inputs.prompts[chunk_idx]

    # Temporal transition
    if chunk_idx in inputs.transitions:
        kwargs["transition"] = inputs.transitions[chunk_idx]

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

    # Noise controller
    if request.noise_controller is not None:
        kwargs["noise_controller"] = request.noise_controller

    # KV cache attention bias
    kv_bias = get_chunk_value(request.kv_cache_attention_bias, chunk_idx)
    if kv_bias is not None:
        kwargs["kv_cache_attention_bias"] = kv_bias

    # Prompt interpolation method
    kwargs["prompt_interpolation_method"] = request.prompt_interpolation_method

    # VACE use input video
    if request.vace_use_input_video is not None:
        kwargs["vace_use_input_video"] = request.vace_use_input_video

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

    # VACE conditioning: per-chunk spec takes priority over global
    logger.info(
        f"build_chunk_kwargs: chunk {chunk_idx}, vace_chunk_specs keys={list(inputs.vace_chunk_specs.keys())}, has_global_frames={inputs.vace_frames is not None}, has_global_masks={inputs.vace_masks is not None}"
    )
    if chunk_idx in inputs.vace_chunk_specs:
        logger.info(f"build_chunk_kwargs: chunk {chunk_idx} USING PER-CHUNK VACE SPEC")
        spec = inputs.vace_chunk_specs[chunk_idx]

        if "frames" in spec:
            frames = spec["frames"]
            frames = pad_chunk(frames, chunk_size, axis=2)
            kwargs["vace_input_frames"] = torch.from_numpy(frames).to(device, dtype)

        if "masks" in spec:
            masks = spec["masks"]
            masks = pad_chunk(masks, chunk_size, axis=2)
            kwargs["vace_input_masks"] = torch.from_numpy(masks).to(device, dtype)

        if "context_scale" in spec:
            kwargs["vace_context_scale"] = spec["context_scale"]
    else:
        logger.info(f"build_chunk_kwargs: chunk {chunk_idx} USING GLOBAL VACE FALLBACK")
        # Global VACE conditioning frames [1, C, T, H, W]
        if inputs.vace_frames is not None:
            chunk = inputs.vace_frames[:, :, start_frame:end_frame, :, :]
            chunk = pad_chunk(chunk, chunk_size, axis=2)
            kwargs["vace_input_frames"] = torch.from_numpy(chunk).to(device, dtype)

        # Global VACE masks [1, 1, T, H, W]
        if inputs.vace_masks is not None:
            chunk = inputs.vace_masks[:, :, start_frame:end_frame, :, :]
            chunk = pad_chunk(chunk, chunk_size, axis=2)
            kwargs["vace_input_masks"] = torch.from_numpy(chunk).to(device, dtype)

    return kwargs


def _log_chunk_info(kwargs: dict, chunk_idx: int, num_chunks: int, logger: "Logger"):
    """Log detailed chunk information."""
    logger.info(f"generate_video_stream: Starting chunk {chunk_idx + 1}/{num_chunks}")
    if kwargs.get("init_cache"):
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Resetting cache (init_cache=True)"
        )
    if "prompts" in kwargs:
        prompt_texts = [p["text"] for p in kwargs["prompts"]]
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Updating prompt to {prompt_texts}"
        )
    if "transition" in kwargs:
        target_texts = [p["text"] for p in kwargs["transition"]["target_prompts"]]
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Temporal transition to {target_texts} "
            f"over {kwargs['transition']['num_steps']} steps "
            f"(method: {kwargs['transition']['temporal_interpolation_method']})"
        )
    if "first_frame_image" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Using first frame keyframe"
        )
    if "last_frame_image" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Using last frame keyframe"
        )
    if "extension_mode" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Extension mode: {kwargs['extension_mode']}"
        )
    if "vace_ref_images" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Using {len(kwargs['vace_ref_images'])} VACE reference images"
        )
    if "vace_input_frames" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: VACE input frames shape: {kwargs['vace_input_frames'].shape}"
        )
    if "vace_input_masks" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: VACE input masks shape: {kwargs['vace_input_masks'].shape}"
        )
    if "vace_context_scale" in kwargs and kwargs["vace_context_scale"] != 1.0:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: VACE context scale: {kwargs['vace_context_scale']}"
        )
    if "vace_use_input_video" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: VACE use input video: {kwargs['vace_use_input_video']}"
        )
    if "video" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Video-to-video mode with {len(kwargs['video'])} frames, noise_scale={kwargs.get('noise_scale', DEFAULT_NOISE_SCALE)}"
        )
    elif "num_frames" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Text-to-video mode generating {kwargs['num_frames']} frames"
        )
    if "denoising_step_list" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Denoising steps: {kwargs['denoising_step_list']}"
        )
    if "noise_controller" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: Using noise controller: {kwargs['noise_controller']}"
        )
    if "kv_cache_attention_bias" in kwargs:
        logger.info(
            f"generate_video_stream: Chunk {chunk_idx}: KV cache attention bias: {kwargs['kv_cache_attention_bias']}"
        )


def _write_chunk_output(
    result: dict,
    chunk_idx: int,
    num_chunks: int,
    chunk_latency: float,
    output_file,
    latency_measures: list,
    fps_measures: list,
    logger: "Logger",
    total_frames_ref: list,
    dimensions_ref: list,
) -> str:
    """Write chunk output to file and return SSE progress event."""
    chunk_output = result["video"]
    num_output_frames = chunk_output.shape[0]
    chunk_fps = num_output_frames / chunk_latency

    latency_measures.append(chunk_latency)
    fps_measures.append(chunk_fps)

    logger.info(
        f"Chunk {chunk_idx + 1}/{num_chunks}: "
        f"{num_output_frames} frames, latency={chunk_latency:.2f}s, fps={chunk_fps:.2f}"
    )

    chunk_np = chunk_output.detach().cpu().numpy()
    chunk_uint8 = (chunk_np * 255).clip(0, 255).astype(np.uint8)
    output_file.write(chunk_uint8.tobytes())

    total_frames_ref[0] += num_output_frames
    if dimensions_ref[0] is None:
        dimensions_ref[0] = chunk_np.shape[1]
        dimensions_ref[1] = chunk_np.shape[2]
        dimensions_ref[2] = chunk_np.shape[3]

    return sse_event(
        "progress",
        {
            "chunk": chunk_idx + 1,
            "total_chunks": num_chunks,
            "frames": num_output_frames,
            "latency": round(chunk_latency, 3),
            "fps": round(chunk_fps, 2),
        },
    )


def _generate_sequential(
    request: "GenerateRequest",
    pipeline,
    inputs: DecodedInputs,
    num_chunks: int,
    chunk_size: int,
    status_info: dict,
    device: torch.device,
    dtype: torch.dtype,
    output_file,
    latency_measures: list,
    fps_measures: list,
    logger: "Logger",
    total_frames_ref: list,
    dimensions_ref: list,
) -> Iterator[str]:
    """Sequential chunk processing (original code path, no processors)."""
    for chunk_idx in range(num_chunks):
        if _cancel_event.is_set():
            logger.info("Generation cancelled by user")
            yield sse_event(
                "cancelled",
                {
                    "chunk": chunk_idx,
                    "total_chunks": num_chunks,
                    "frames_completed": total_frames_ref[0],
                },
            )
            return

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
        _log_chunk_info(kwargs, chunk_idx, num_chunks, logger)

        chunk_start = time.time()
        with torch.amp.autocast("cuda", dtype=dtype):
            result = pipeline(**kwargs)
        chunk_latency = time.time() - chunk_start

        yield _write_chunk_output(
            result,
            chunk_idx,
            num_chunks,
            chunk_latency,
            output_file,
            latency_measures,
            fps_measures,
            logger,
            total_frames_ref,
            dimensions_ref,
        )


def _generate_with_processors(
    request: "GenerateRequest",
    pipeline,
    pipeline_manager: "PipelineManager",
    inputs: DecodedInputs,
    num_chunks: int,
    chunk_size: int,
    status_info: dict,
    device: torch.device,
    dtype: torch.dtype,
    output_file,
    latency_measures: list,
    fps_measures: list,
    logger: "Logger",
    total_frames_ref: list,
    dimensions_ref: list,
) -> Iterator[str]:
    """Chunk processing with pre/post processor pipeline chaining."""
    from .pipeline_processor import _SENTINEL, PipelineProcessor

    # Build the processor chain
    processors: list[PipelineProcessor] = []

    if request.pre_processor_id:
        pre_pipeline = pipeline_manager.get_pipeline_by_id(request.pre_processor_id)
        pre_proc = PipelineProcessor(
            pipeline=pre_pipeline,
            pipeline_id=request.pre_processor_id,
            batch_mode=True,
        )
        processors.append(pre_proc)
        logger.info(f"Pre-processor: {request.pre_processor_id}")

    main_proc = PipelineProcessor(
        pipeline=pipeline,
        pipeline_id=request.pipeline_id,
        batch_mode=True,
    )
    processors.append(main_proc)

    if request.post_processor_id:
        post_pipeline = pipeline_manager.get_pipeline_by_id(request.post_processor_id)
        post_proc = PipelineProcessor(
            pipeline=post_pipeline,
            pipeline_id=request.post_processor_id,
            batch_mode=True,
        )
        processors.append(post_proc)
        logger.info(f"Post-processor: {request.post_processor_id}")

    # Chain processors
    for i in range(len(processors) - 1):
        processors[i].set_next_processor(processors[i + 1])

    # Start all processors
    for proc in processors:
        proc.start()

    first_proc = processors[0]
    last_proc = processors[-1]

    try:
        # Feed chunks into the first processor's input queue
        for chunk_idx in range(num_chunks):
            if _cancel_event.is_set():
                logger.info("Generation cancelled by user")
                yield sse_event(
                    "cancelled",
                    {
                        "chunk": chunk_idx,
                        "total_chunks": num_chunks,
                        "frames_completed": total_frames_ref[0],
                    },
                )
                return

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
            _log_chunk_info(kwargs, chunk_idx, num_chunks, logger)

            chunk_start = time.time()

            # Feed kwargs into chain (blocking put)
            first_proc.input_queue.put(kwargs)

            # Collect result from last processor (blocking get)
            while True:
                try:
                    result = last_proc.output_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    if _cancel_event.is_set():
                        return
                    continue

            chunk_latency = time.time() - chunk_start

            yield _write_chunk_output(
                result,
                chunk_idx,
                num_chunks,
                chunk_latency,
                output_file,
                latency_measures,
                fps_measures,
                logger,
                total_frames_ref,
                dimensions_ref,
            )

        # Signal end of input
        first_proc.input_queue.put(_SENTINEL)

    finally:
        # Stop all processors
        for proc in processors:
            proc.stop()


def generate_video_stream(
    request: "GenerateRequest",
    pipeline_manager: "PipelineManager",
    status_info: dict,
    logger: "Logger",
) -> Iterator[str]:
    """Generate video frames, yielding SSE events.

    Writes output to temp file incrementally, returns output_path for download.
    """
    _cancel_event.clear()
    output_file_path = None
    completed = False

    try:
        pipeline = pipeline_manager.get_pipeline_by_id(request.pipeline_id)

        # Determine chunk size from pipeline
        has_video = request.input_video is not None or request.input_path is not None
        requirements = pipeline.prepare(video=[] if has_video else None)
        chunk_size = requirements.input_size if requirements else DEFAULT_CHUNK_SIZE
        num_chunks = (request.num_frames + chunk_size - 1) // chunk_size

        # Decode inputs (supports both file-based and base64)
        inputs = decode_inputs(request, request.num_frames, logger)

        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16
        latency_measures = []
        fps_measures = []

        # Create output file for incremental writing (reuse recording pattern)
        from .recording import TEMP_FILE_PREFIXES, RecordingManager

        output_file_path = RecordingManager._create_temp_file(
            ".bin", TEMP_FILE_PREFIXES["generate_output"]
        )
        output_file = open(output_file_path, "wb")

        # We'll write a placeholder header, then update it at the end
        # Header format: ndim (4 bytes) + shape (4 * ndim bytes)
        # For video [T, H, W, C], that's 4 + 16 = 20 bytes
        header_size = 4 + 4 * 4  # ndim + 4 dimensions
        output_file.write(b"\x00" * header_size)  # Placeholder

        total_frames = 0
        video_height = None
        video_width = None
        video_channels = None

        # Determine if we need processor chaining
        use_processors = (
            request.pre_processor_id is not None
            or request.post_processor_id is not None
        )

        try:
            if use_processors:
                yield from _generate_with_processors(
                    request,
                    pipeline,
                    pipeline_manager,
                    inputs,
                    num_chunks,
                    chunk_size,
                    status_info,
                    device,
                    dtype,
                    output_file,
                    latency_measures,
                    fps_measures,
                    logger,
                    _total_frames_ref := [0],
                    _dimensions_ref := [None, None, None],
                )
                total_frames = _total_frames_ref[0]
                video_height, video_width, video_channels = _dimensions_ref
            else:
                yield from _generate_sequential(
                    request,
                    pipeline,
                    inputs,
                    num_chunks,
                    chunk_size,
                    status_info,
                    device,
                    dtype,
                    output_file,
                    latency_measures,
                    fps_measures,
                    logger,
                    _total_frames_ref := [0],
                    _dimensions_ref := [None, None, None],
                )
                total_frames = _total_frames_ref[0]
                video_height, video_width, video_channels = _dimensions_ref

            # Update header with actual shape
            output_file.seek(0)
            shape = (total_frames, video_height, video_width, video_channels)
            output_file.write(len(shape).to_bytes(4, "little"))
            for dim in shape:
                output_file.write(dim.to_bytes(4, "little"))

        finally:
            output_file.close()

        logger.info(f"Output video saved: {output_file_path}")

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

        output_shape = [total_frames, video_height, video_width, video_channels]

        yield sse_event(
            "complete",
            {
                "output_path": output_file_path,
                "video_shape": output_shape,
                "num_frames": total_frames,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
            },
        )
        completed = True

    except Exception as e:
        logger.exception("Error generating video")
        yield sse_event("error", {"error": str(e)})

    finally:
        # Clean up uploaded input file
        if request.input_path:
            try:
                Path(request.input_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up input file: {request.input_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up input file: {e}")

        # Clean up output file if generation didn't complete successfully
        if not completed and output_file_path:
            try:
                Path(output_file_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up orphaned output file: {output_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up output file: {e}")
