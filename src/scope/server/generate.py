"""Video generation service for batch mode with chunked processing."""

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
    from .schema import ChunkSpec, GenerateRequest


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


def sse_event(event_type: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@dataclass
class DecodedInputs:
    """Decoded and preprocessed inputs for generation."""

    input_video: np.ndarray | None = None
    first_frames: dict[int, str] = field(default_factory=dict)
    last_frames: dict[int, str] = field(default_factory=dict)
    ref_images: dict[int, list[str]] = field(default_factory=dict)
    prompts: dict[int, list[dict]] = field(default_factory=dict)
    transitions: dict[int, dict] = field(default_factory=dict)
    vace_chunk_specs: dict[int, dict] = field(default_factory=dict)
    input_video_chunks: dict[int, np.ndarray] = field(default_factory=dict)
    chunk_specs_map: "dict[int, ChunkSpec]" = field(default_factory=dict)


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
    """Decode all inputs from request using unified ChunkSpec."""
    inputs = DecodedInputs()

    # Input video from file path
    if request.input_path:
        logger.info(f"Loading input video from file: {request.input_path}")
        inputs.input_video = load_video_from_file(request.input_path)
        inputs.input_video = loop_to_length(inputs.input_video, num_frames, axis=0)

    # Default prompt
    if isinstance(request.prompt, str):
        inputs.prompts = {0: [{"text": request.prompt, "weight": PROMPT_WEIGHT}]}
    else:
        inputs.prompts = {
            0: [{"text": p.text, "weight": p.weight} for p in request.prompt]
        }

    # Load binary blob if provided
    blob: bytes | None = None
    if request.data_blob_path:
        import tempfile

        from .recording import TEMP_FILE_PREFIXES

        # Security: validate path prefix and temp dir
        blob_path = Path(request.data_blob_path)
        temp_dir = Path(tempfile.gettempdir())
        if not blob_path.is_relative_to(temp_dir) or not blob_path.name.startswith(
            TEMP_FILE_PREFIXES["generate_data"]
        ):
            raise ValueError(
                f"Invalid data_blob_path: must be a temp file with prefix {TEMP_FILE_PREFIXES['generate_data']}"
            )
        with open(blob_path, "rb") as f:
            blob = f.read()
        logger.info(
            f"decode_inputs: Loaded data blob from {request.data_blob_path} ({len(blob)} bytes)"
        )

    # Process chunk specs — single loop, single source of truth
    for spec in request.chunk_specs or []:
        # Store spec for build_chunk_kwargs
        inputs.chunk_specs_map[spec.chunk] = spec

        # Prompts
        if spec.prompts:
            inputs.prompts[spec.chunk] = [
                {"text": p.text, "weight": p.weight} for p in spec.prompts
            ]
        elif spec.text:
            inputs.prompts[spec.chunk] = [{"text": spec.text, "weight": PROMPT_WEIGHT}]

        # Transitions
        if spec.transition_target_prompts:
            inputs.transitions[spec.chunk] = {
                "target_prompts": [
                    {"text": p.text, "weight": p.weight}
                    for p in spec.transition_target_prompts
                ],
                "num_steps": spec.transition_num_steps or 4,
                "temporal_interpolation_method": spec.transition_method or "linear",
            }

        # Keyframes
        if spec.first_frame_image:
            inputs.first_frames[spec.chunk] = spec.first_frame_image
        if spec.last_frame_image:
            inputs.last_frames[spec.chunk] = spec.last_frame_image
        if spec.vace_ref_images:
            inputs.ref_images[spec.chunk] = spec.vace_ref_images

        # VACE from blob
        if blob is not None and spec.vace_frames_offset is not None:
            decoded: dict = {"vace_temporally_locked": spec.vace_temporally_locked}
            if spec.vace_frames_shape and spec.vace_frames_offset is not None:
                count = 1
                for d in spec.vace_frames_shape:
                    count *= d
                arr = np.frombuffer(
                    blob, dtype=np.float32, count=count, offset=spec.vace_frames_offset
                ).reshape(spec.vace_frames_shape)
                decoded["frames"] = arr
                logger.info(
                    f"decode_inputs: chunk {spec.chunk} VACE frames shape={arr.shape}"
                )
            if spec.vace_masks_shape and spec.vace_masks_offset is not None:
                count = 1
                for d in spec.vace_masks_shape:
                    count *= d
                arr = np.frombuffer(
                    blob, dtype=np.float32, count=count, offset=spec.vace_masks_offset
                ).reshape(spec.vace_masks_shape)
                decoded["masks"] = arr
                logger.info(
                    f"decode_inputs: chunk {spec.chunk} VACE masks shape={arr.shape}"
                )
            if spec.vace_context_scale is not None:
                decoded["context_scale"] = spec.vace_context_scale
            inputs.vace_chunk_specs[spec.chunk] = decoded

        # Input video from blob (per-chunk video-to-video)
        if (
            blob is not None
            and spec.input_video_offset is not None
            and spec.input_video_shape is not None
        ):
            count = 1
            for d in spec.input_video_shape:
                count *= d
            inputs.input_video_chunks[spec.chunk] = np.frombuffer(
                blob, dtype=np.uint8, count=count, offset=spec.input_video_offset
            ).reshape(spec.input_video_shape)

    logger.info(
        f"decode_inputs: prompts={list(inputs.prompts.keys())}, "
        f"transitions={list(inputs.transitions.keys())}, "
        f"vace_specs={list(inputs.vace_chunk_specs.keys())}, "
        f"input_video_chunks={list(inputs.input_video_chunks.keys())}, "
        f"first_frames={list(inputs.first_frames.keys())}, "
        f"last_frames={list(inputs.last_frames.keys())}"
    )

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
    """Build pipeline kwargs for a single chunk.

    Per-chunk ChunkSpec values override request-level globals.
    """
    # Get per-chunk spec (if any)
    spec = inputs.chunk_specs_map.get(chunk_idx)

    kwargs = {
        "height": request.height
        or status_info.get("load_params", {}).get("height", DEFAULT_HEIGHT),
        "width": request.width
        or status_info.get("load_params", {}).get("width", DEFAULT_WIDTH),
        "base_seed": spec.seed if spec and spec.seed is not None else request.seed,
        "init_cache": chunk_idx == 0 or (spec is not None and spec.reset_cache),
        "manage_cache": (
            spec.manage_cache
            if spec and spec.manage_cache is not None
            else request.manage_cache
        ),
    }

    # Prompt (sticky behavior - only send when it changes)
    if chunk_idx in inputs.prompts:
        kwargs["prompts"] = inputs.prompts[chunk_idx]

    # Temporal transition
    if chunk_idx in inputs.transitions:
        kwargs["transition"] = inputs.transitions[chunk_idx]

    if request.denoising_steps:
        kwargs["denoising_step_list"] = request.denoising_steps

    # Video-to-video: per-chunk input video takes priority over global input video
    if chunk_idx in inputs.input_video_chunks:
        # Per-chunk input video from blob (enables v2v/t2v switching per chunk)
        chunk_frames = inputs.input_video_chunks[chunk_idx]
        chunk_frames = pad_chunk(chunk_frames, chunk_size, axis=0)
        kwargs["video"] = [torch.from_numpy(f).unsqueeze(0) for f in chunk_frames]
        kwargs["noise_scale"] = (
            spec.noise_scale
            if spec and spec.noise_scale is not None
            else request.noise_scale
        )
        logger.info(
            f"Chunk {chunk_idx}: Using per-chunk input video ({chunk_frames.shape[0]} frames)"
        )
    elif inputs.input_video is not None:
        chunk_frames = inputs.input_video[start_frame:end_frame]
        chunk_frames = pad_chunk(chunk_frames, chunk_size, axis=0)
        kwargs["video"] = [torch.from_numpy(f).unsqueeze(0) for f in chunk_frames]
        kwargs["noise_scale"] = (
            spec.noise_scale
            if spec and spec.noise_scale is not None
            else request.noise_scale
        )
    else:
        kwargs["num_frames"] = chunk_size

    # VACE context scale
    kwargs["vace_context_scale"] = (
        spec.vace_context_scale
        if spec and spec.vace_context_scale is not None
        else request.vace_context_scale
    )

    # Noise controller
    noise_ctrl = (
        spec.noise_controller
        if spec and spec.noise_controller is not None
        else request.noise_controller
    )
    if noise_ctrl is not None:
        kwargs["noise_controller"] = noise_ctrl

    # KV cache attention bias
    kv_bias = (
        spec.kv_cache_attention_bias
        if spec and spec.kv_cache_attention_bias is not None
        else request.kv_cache_attention_bias
    )
    if kv_bias is not None:
        kwargs["kv_cache_attention_bias"] = kv_bias

    # Prompt interpolation method
    kwargs["prompt_interpolation_method"] = (
        spec.prompt_interpolation_method
        if spec and spec.prompt_interpolation_method is not None
        else request.prompt_interpolation_method
    )

    # VACE use input video
    if request.vace_use_input_video is not None:
        kwargs["vace_use_input_video"] = request.vace_use_input_video

    # LoRA scales: per-chunk spec overrides global
    lora_scales = spec.lora_scales if spec and spec.lora_scales else request.lora_scales
    if lora_scales:
        lora_scale_updates = []
        for path, scale in lora_scales.items():
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

    # VACE conditioning from blob
    logger.info(
        f"build_chunk_kwargs: chunk {chunk_idx}, vace_chunk_specs keys={list(inputs.vace_chunk_specs.keys())}"
    )
    if chunk_idx in inputs.vace_chunk_specs:
        logger.info(f"build_chunk_kwargs: chunk {chunk_idx} USING PER-CHUNK VACE SPEC")
        vace_spec = inputs.vace_chunk_specs[chunk_idx]

        if "frames" in vace_spec:
            frames = vace_spec["frames"]
            frames = pad_chunk(frames, chunk_size, axis=2)
            kwargs["vace_input_frames"] = torch.from_numpy(frames).to(device, dtype)

        if "masks" in vace_spec:
            masks = vace_spec["masks"]
            masks = pad_chunk(masks, chunk_size, axis=2)
            kwargs["vace_input_masks"] = torch.from_numpy(masks).to(device, dtype)

        if "context_scale" in vace_spec:
            kwargs["vace_context_scale"] = vace_spec["context_scale"]
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
        has_video = request.input_path is not None
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
        # Clean up uploaded data blob file
        if request.data_blob_path:
            try:
                Path(request.data_blob_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up data blob file: {request.data_blob_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up data blob file: {e}")

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
