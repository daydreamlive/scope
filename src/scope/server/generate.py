"""Video generation service for batch mode with chunked processing."""

import concurrent.futures
import json
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING

import numpy as np
import torch

# Cancellation support (single-client, so one event suffices)
_cancel_event = threading.Event()

# Generation lock (single-client: only one generation at a time)
_generation_lock = threading.Lock()

# Max data blob upload size (2 GB)
MAX_DATA_BLOB_BYTES = 2 * 1024 * 1024 * 1024


def cancel_generation():
    """Signal the current generation to stop after the current chunk."""
    _cancel_event.set()


def is_generation_cancelled() -> bool:
    """Check if cancellation has been requested."""
    return _cancel_event.is_set()


def is_generation_active() -> bool:
    """Check if a generation is currently in progress."""
    return _generation_lock.locked()


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


# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def sse_event(event_type: str, data: dict) -> str:
    """Format a server-sent event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


@dataclass
class GenerationState:
    """Mutable state accumulated during chunk-by-chunk generation."""

    output_file: IO[bytes]
    num_chunks: int
    logger: "Logger"
    total_frames: int = 0
    height: int | None = None
    width: int | None = None
    channels: int | None = None
    latencies: list[float] = field(default_factory=list)
    fps_measures: list[float] = field(default_factory=list)

    def build_chunk_sse(self, chunk_idx: int, chunk_latency: float) -> str:
        """Build SSE progress event (call from main thread before write)."""
        return sse_event(
            "progress",
            {
                "chunk": chunk_idx + 1,
                "total_chunks": self.num_chunks,
                "latency": round(chunk_latency, 3),
            },
        )

    def write_chunk(self, result: dict, chunk_idx: int, chunk_latency: float) -> None:
        """Write chunk output to file (safe to call from background thread)."""
        chunk_output = result["video"]
        num_output_frames = chunk_output.shape[0]
        chunk_fps = num_output_frames / chunk_latency

        self.latencies.append(chunk_latency)
        self.fps_measures.append(chunk_fps)

        self.logger.info(
            f"Chunk {chunk_idx + 1}/{self.num_chunks}: "
            f"{num_output_frames} frames, latency={chunk_latency:.2f}s, fps={chunk_fps:.2f}"
        )

        chunk_np = chunk_output.detach().cpu().numpy()
        chunk_uint8 = (chunk_np * 255).clip(0, 255).astype(np.uint8)
        self.output_file.write(chunk_uint8.tobytes())

        self.total_frames += num_output_frames
        if self.height is None:
            self.height = chunk_np.shape[1]
            self.width = chunk_np.shape[2]
            self.channels = chunk_np.shape[3]

    @property
    def output_shape(self) -> list[int]:
        return [self.total_frames, self.height, self.width, self.channels]

    def log_summary(self):
        """Log performance summary."""
        if not self.latencies:
            return
        avg_lat = sum(self.latencies) / len(self.latencies)
        avg_fps = sum(self.fps_measures) / len(self.fps_measures)
        self.logger.info(
            f"=== Performance Summary ({self.num_chunks} chunks) ===\n"
            f"  Latency - Avg: {avg_lat:.2f}s, "
            f"Max: {max(self.latencies):.2f}s, Min: {min(self.latencies):.2f}s\n"
            f"  FPS - Avg: {avg_fps:.2f}, "
            f"Max: {max(self.fps_measures):.2f}, Min: {min(self.fps_measures):.2f}"
        )


# ---------------------------------------------------------------------------
# Input decoding
# ---------------------------------------------------------------------------


def load_video_from_file(file_path: str) -> np.ndarray:
    """Load video from temp file with header (ndim + shape + raw uint8)."""
    with open(file_path, "rb") as f:
        ndim = int.from_bytes(f.read(4), "little")
        shape = tuple(int.from_bytes(f.read(4), "little") for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data


def _read_blob_array(
    blob: bytes, offset: int, shape: list[int], dtype=np.float32
) -> np.ndarray:
    """Read a contiguous array from a binary blob at a given offset."""
    count = 1
    for d in shape:
        count *= d
    return np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape(shape)


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
                arr = _read_blob_array(
                    blob, spec.vace_frames_offset, spec.vace_frames_shape
                )
                decoded["frames"] = arr
                logger.info(
                    f"decode_inputs: chunk {spec.chunk} VACE frames shape={arr.shape}"
                )
            if spec.vace_masks_shape and spec.vace_masks_offset is not None:
                arr = _read_blob_array(
                    blob, spec.vace_masks_offset, spec.vace_masks_shape
                )
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
            inputs.input_video_chunks[spec.chunk] = _read_blob_array(
                blob, spec.input_video_offset, spec.input_video_shape, dtype=np.uint8
            )

    logger.info(
        f"decode_inputs: prompts={list(inputs.prompts.keys())}, "
        f"transitions={list(inputs.transitions.keys())}, "
        f"vace_specs={list(inputs.vace_chunk_specs.keys())}, "
        f"input_video_chunks={list(inputs.input_video_chunks.keys())}, "
        f"first_frames={list(inputs.first_frames.keys())}, "
        f"last_frames={list(inputs.last_frames.keys())}"
    )

    return inputs


# ---------------------------------------------------------------------------
# Chunk kwargs builder
# ---------------------------------------------------------------------------


def _resolve(spec, attr: str, request, fallback=None):
    """Return per-chunk spec value if set, else request-level value, else fallback."""
    if spec is not None:
        val = getattr(spec, attr, None)
        if val is not None:
            return val
    val = getattr(request, attr, None)
    return val if val is not None else fallback


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
    spec = inputs.chunk_specs_map.get(chunk_idx)
    load_params = status_info.get("load_params", {})

    kwargs: dict = {
        "height": request.height
        if request.height is not None
        else load_params.get("height", DEFAULT_HEIGHT),
        "width": request.width
        if request.width is not None
        else load_params.get("width", DEFAULT_WIDTH),
        "base_seed": _resolve(spec, "seed", request, DEFAULT_SEED),
        "init_cache": chunk_idx == 0 or (spec is not None and spec.reset_cache),
        "manage_cache": _resolve(spec, "manage_cache", request, True),
    }

    # Prompt (sticky — only send when it changes)
    if chunk_idx in inputs.prompts:
        kwargs["prompts"] = inputs.prompts[chunk_idx]

    # Temporal transition
    if chunk_idx in inputs.transitions:
        kwargs["transition"] = inputs.transitions[chunk_idx]

    if request.denoising_steps:
        kwargs["denoising_step_list"] = request.denoising_steps

    # Video-to-video: per-chunk input video takes priority over global
    if chunk_idx in inputs.input_video_chunks:
        chunk_frames = pad_chunk(
            inputs.input_video_chunks[chunk_idx], chunk_size, axis=0
        )
        kwargs["video"] = [torch.from_numpy(f).unsqueeze(0) for f in chunk_frames]
        kwargs["noise_scale"] = _resolve(
            spec, "noise_scale", request, DEFAULT_NOISE_SCALE
        )
        logger.info(
            f"Chunk {chunk_idx}: Using per-chunk input video ({chunk_frames.shape[0]} frames)"
        )
    elif inputs.input_video is not None:
        chunk_frames = pad_chunk(
            inputs.input_video[start_frame:end_frame], chunk_size, axis=0
        )
        kwargs["video"] = [torch.from_numpy(f).unsqueeze(0) for f in chunk_frames]
        kwargs["noise_scale"] = _resolve(
            spec, "noise_scale", request, DEFAULT_NOISE_SCALE
        )
    else:
        kwargs["num_frames"] = chunk_size

    kwargs["vace_context_scale"] = _resolve(spec, "vace_context_scale", request, 1.0)
    kwargs["prompt_interpolation_method"] = _resolve(
        spec, "prompt_interpolation_method", request, "linear"
    )

    # Optional overrides (only include in kwargs when non-None)
    noise_ctrl = _resolve(spec, "noise_controller", request)
    if noise_ctrl is not None:
        kwargs["noise_controller"] = noise_ctrl

    kv_bias = _resolve(spec, "kv_cache_attention_bias", request)
    if kv_bias is not None:
        kwargs["kv_cache_attention_bias"] = kv_bias

    if request.vace_use_input_video is not None:
        kwargs["vace_use_input_video"] = request.vace_use_input_video

    # LoRA scales: per-chunk spec overrides global
    lora_scales = spec.lora_scales if spec and spec.lora_scales else request.lora_scales
    if lora_scales:
        kwargs["lora_scales"] = [
            {"path": p, "scale": s} for p, s in lora_scales.items()
        ]
        for p, s in lora_scales.items():
            logger.info(f"Chunk {chunk_idx}: LoRA scale={s:.3f} for {Path(p).name}")

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
    if chunk_idx in inputs.vace_chunk_specs:
        vace_spec = inputs.vace_chunk_specs[chunk_idx]
        if "frames" in vace_spec:
            frames = pad_chunk(vace_spec["frames"], chunk_size, axis=2)
            kwargs["vace_input_frames"] = torch.from_numpy(frames).to(device, dtype)
        if "masks" in vace_spec:
            masks = pad_chunk(vace_spec["masks"], chunk_size, axis=2)
            kwargs["vace_input_masks"] = torch.from_numpy(masks).to(device, dtype)
        if "context_scale" in vace_spec:
            kwargs["vace_context_scale"] = vace_spec["context_scale"]

    return kwargs


# ---------------------------------------------------------------------------
# Chunk logging
# ---------------------------------------------------------------------------

# (key, format_string) — format_string uses {v} for the value
_CHUNK_LOG_ENTRIES = [
    ("init_cache", "Resetting cache (init_cache=True)", lambda v: v),
    ("extension_mode", "Extension mode: {v}", None),
    ("vace_context_scale", "VACE context scale: {v}", lambda v: v != 1.0),
    ("vace_use_input_video", "VACE use input video: {v}", None),
    ("denoising_step_list", "Denoising steps: {v}", None),
    ("noise_controller", "Using noise controller: {v}", None),
    ("kv_cache_attention_bias", "KV cache attention bias: {v}", None),
]


def _log_chunk_info(kwargs: dict, chunk_idx: int, num_chunks: int, logger: "Logger"):
    """Log detailed chunk information."""
    prefix = f"generate: Chunk {chunk_idx}"
    logger.info(f"generate: Starting chunk {chunk_idx + 1}/{num_chunks}")

    # Structured entries
    if "prompts" in kwargs:
        logger.info(f"{prefix}: Prompt → {[p['text'] for p in kwargs['prompts']]}")
    if "transition" in kwargs:
        t = kwargs["transition"]
        logger.info(
            f"{prefix}: Transition → {[p['text'] for p in t['target_prompts']]} "
            f"over {t['num_steps']} steps ({t['temporal_interpolation_method']})"
        )
    if "first_frame_image" in kwargs:
        logger.info(f"{prefix}: Using first frame keyframe")
    if "last_frame_image" in kwargs:
        logger.info(f"{prefix}: Using last frame keyframe")
    if "vace_ref_images" in kwargs:
        logger.info(
            f"{prefix}: Using {len(kwargs['vace_ref_images'])} VACE reference images"
        )
    if "vace_input_frames" in kwargs:
        logger.info(
            f"{prefix}: VACE input frames shape: {kwargs['vace_input_frames'].shape}"
        )
    if "vace_input_masks" in kwargs:
        logger.info(
            f"{prefix}: VACE input masks shape: {kwargs['vace_input_masks'].shape}"
        )
    if "video" in kwargs:
        logger.info(
            f"{prefix}: Video-to-video ({len(kwargs['video'])} frames, "
            f"noise_scale={kwargs.get('noise_scale', DEFAULT_NOISE_SCALE)})"
        )
    elif "num_frames" in kwargs:
        logger.info(f"{prefix}: Text-to-video ({kwargs['num_frames']} frames)")

    # Table-driven simple entries
    for key, msg, condition in _CHUNK_LOG_ENTRIES:
        if key in kwargs:
            v = kwargs[key]
            if condition is None or condition(v):
                logger.info(f"{prefix}: {msg.format(v=v)}")


# ---------------------------------------------------------------------------
# Generation engine
# ---------------------------------------------------------------------------


def _generate_chunks(
    request: "GenerateRequest",
    pipeline,
    pipeline_manager: "PipelineManager",
    inputs: DecodedInputs,
    num_chunks: int,
    chunk_size: int,
    status_info: dict,
    device: torch.device,
    dtype: torch.dtype,
    state: GenerationState,
    logger: "Logger",
) -> Iterator[str]:
    """Process chunks through a processor chain, yielding SSE events.

    Always uses PipelineProcessor — when there are no pre/post processors
    the chain is just [main_pipeline].
    """
    from .pipeline_processor import _SENTINEL, PipelineProcessor

    # Build processor chain: [pre?] → main → [post?]
    processors: list[PipelineProcessor] = []

    if request.pre_processor_id:
        pre_pipeline = pipeline_manager.get_pipeline_by_id(request.pre_processor_id)
        processors.append(
            PipelineProcessor(
                pipeline=pre_pipeline,
                pipeline_id=request.pre_processor_id,
                batch_mode=True,
            )
        )
        logger.info(f"Pre-processor: {request.pre_processor_id}")

    processors.append(
        PipelineProcessor(
            pipeline=pipeline, pipeline_id=request.pipeline_id, batch_mode=True
        )
    )

    if request.post_processor_id:
        post_pipeline = pipeline_manager.get_pipeline_by_id(request.post_processor_id)
        processors.append(
            PipelineProcessor(
                pipeline=post_pipeline,
                pipeline_id=request.post_processor_id,
                batch_mode=True,
            )
        )
        logger.info(f"Post-processor: {request.post_processor_id}")

    # Chain and start
    for i in range(len(processors) - 1):
        processors[i].set_next_processor(processors[i + 1])
    for proc in processors:
        proc.start()

    first_proc = processors[0]
    last_proc = processors[-1]

    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    write_future: concurrent.futures.Future | None = None
    pending_sse: str | None = None

    try:
        for chunk_idx in range(num_chunks):
            if _cancel_event.is_set():
                logger.info("Generation cancelled by user")
                yield sse_event(
                    "cancelled",
                    {
                        "chunk": chunk_idx,
                        "total_chunks": num_chunks,
                        "frames_completed": state.total_frames,
                    },
                )
                return

            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, request.num_frames)

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

            first_proc.input_queue.put(kwargs)

            # Collect result from last processor
            while True:
                try:
                    result = last_proc.output_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    if _cancel_event.is_set():
                        return
                    continue

            chunk_latency = time.time() - chunk_start

            # Wait for previous async write before starting a new one
            if write_future is not None:
                write_future.result()
            if pending_sse is not None:
                yield pending_sse

            # Offload CPU transfer + disk I/O to background thread
            pending_sse = state.build_chunk_sse(chunk_idx, chunk_latency)
            write_future = write_executor.submit(
                state.write_chunk, result, chunk_idx, chunk_latency
            )

        # Wait for final write
        if write_future is not None:
            write_future.result()
        if pending_sse is not None:
            yield pending_sse

        # Signal end of input
        first_proc.input_queue.put(_SENTINEL)

    finally:
        write_executor.shutdown(wait=True)
        for proc in processors:
            proc.stop()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_video_stream(
    request: "GenerateRequest",
    pipeline_manager: "PipelineManager",
    status_info: dict,
    logger: "Logger",
) -> Iterator[str]:
    """Generate video frames, yielding SSE events.

    Writes output to temp file incrementally, returns output_path for download.
    Only one generation can run at a time (single-client).
    """
    if not _generation_lock.acquire(blocking=False):
        yield sse_event("error", {"error": "A generation is already in progress"})
        return

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

        inputs = decode_inputs(request, request.num_frames, logger)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        # Create output file with placeholder header
        from .recording import TEMP_FILE_PREFIXES, RecordingManager

        output_file_path = RecordingManager._create_temp_file(
            ".bin", TEMP_FILE_PREFIXES["generate_output"]
        )
        output_file = open(output_file_path, "wb")

        # Header: ndim (4 bytes) + shape (4 * ndim bytes) = 20 bytes for [T, H, W, C]
        header_size = 4 + 4 * 4
        output_file.write(b"\x00" * header_size)

        state = GenerationState(
            output_file=output_file, num_chunks=num_chunks, logger=logger
        )

        try:
            yield from _generate_chunks(
                request,
                pipeline,
                pipeline_manager,
                inputs,
                num_chunks,
                chunk_size,
                status_info,
                device,
                dtype,
                state,
                logger,
            )

            # Update header with actual shape
            output_file.seek(0)
            shape = tuple(state.output_shape)
            output_file.write(len(shape).to_bytes(4, "little"))
            for dim in shape:
                output_file.write(dim.to_bytes(4, "little"))

        finally:
            output_file.close()

        logger.info(f"Output video saved: {output_file_path}")
        state.log_summary()

        yield sse_event(
            "complete",
            {
                "output_path": output_file_path,
                "video_shape": state.output_shape,
                "num_frames": state.total_frames,
                "num_chunks": num_chunks,
                "chunk_size": chunk_size,
            },
        )
        completed = True

    except Exception as e:
        logger.exception("Error generating video")
        yield sse_event("error", {"error": str(e)})

    finally:
        # Clean up uploaded files
        for path_attr in ("data_blob_path", "input_path"):
            path = getattr(request, path_attr, None)
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                    logger.info(f"Cleaned up {path_attr}: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {path_attr}: {e}")

        # Clean up output file if generation didn't complete
        if not completed and output_file_path:
            try:
                Path(output_file_path).unlink(missing_ok=True)
                logger.info(f"Cleaned up orphaned output file: {output_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up output file: {e}")

        _generation_lock.release()
