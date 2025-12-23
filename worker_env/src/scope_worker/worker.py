"""Pipeline Worker - Runs pipeline and FrameProcessor in a separate process using ZMQ for IPC."""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from enum import Enum

import numpy as np
import torch
import zmq
from omegaconf import OmegaConf

# Configure logging for worker process
logger = logging.getLogger(__name__)


class WorkerCommand(Enum):
    """Commands that can be sent to the worker process."""

    LOAD_PIPELINE = "load_pipeline"
    UNLOAD_PIPELINE = "unload_pipeline"
    CREATE_FRAME_PROCESSOR = "create_frame_processor"
    DESTROY_FRAME_PROCESSOR = "destroy_frame_processor"
    PUT_FRAME = "put_frame"
    GET_FRAME = "get_frame"
    UPDATE_PARAMETERS = "update_parameters"
    GET_FPS = "get_fps"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    """Response types from worker process."""

    SUCCESS = "success"
    ERROR = "error"
    PIPELINE_LOADED = "pipeline_loaded"
    PIPELINE_NOT_LOADED = "pipeline_not_loaded"
    RESULT = "result"
    FRAME_PROCESSOR_CREATED = "frame_processor_created"
    FRAME = "frame"


# Constants for FrameProcessor
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3
MIN_FPS = 1.0
MAX_FPS = 60.0
DEFAULT_FPS = 30.0
SLEEP_TIME = 0.01


class WorkerFrameProcessor:
    """FrameProcessor that runs in the worker process and uses pipeline directly."""

    def __init__(
        self,
        pipeline,
        max_output_queue_size: int = 8,
        max_parameter_queue_size: int = 8,
        max_buffer_size: int = 30,
        initial_parameters: dict = None,
    ):
        self.pipeline = pipeline

        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.frame_buffer_lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=max_output_queue_size)

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=max_parameter_queue_size)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # FPS tracking variables
        self.processing_time_per_frame = deque(maxlen=2)
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5
        self.min_fps = MIN_FPS
        self.max_fps = MAX_FPS
        self.current_pipeline_fps = DEFAULT_FPS
        self.fps_lock = threading.Lock()

        self.paused = False

    def start(self):
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("WorkerFrameProcessor started")

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        with self.frame_buffer_lock:
            self.frame_buffer.clear()

        logger.info("WorkerFrameProcessor stopped")

    def put(self, frame_data: dict) -> bool:
        """Put a frame into the buffer. frame_data is a serialized VideoFrame."""
        if not self.running:
            return False

        # Deserialize frame from dict
        frame_array = frame_data.get("array")
        if frame_array is None:
            return False

        with self.frame_buffer_lock:
            # Store as dict for now, will convert to tensor when processing
            self.frame_buffer.append(frame_data)
            return True

    def get(self) -> dict | None:
        """Get a processed frame. Returns serialized tensor data."""
        if not self.running:
            return None

        try:
            frame_tensor = self.output_queue.get_nowait()
            # Serialize tensor to dict for inter-process communication
            return {"__tensor__": True, "data": frame_tensor.cpu().numpy()}
        except queue.Empty:
            return None

    def get_current_pipeline_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS"""
        with self.fps_lock:
            return self.current_pipeline_fps

    def _calculate_pipeline_fps(self, start_time: float, num_frames: int):
        """Calculate FPS based on processing time and number of frames created"""
        processing_time = time.time() - start_time
        if processing_time <= 0 or num_frames <= 0:
            return

        time_per_frame = processing_time / num_frames
        self.processing_time_per_frame.append(time_per_frame)

        current_time = time.time()
        if current_time - self.last_fps_update >= self.fps_update_interval:
            if len(self.processing_time_per_frame) >= 1:
                avg_time_per_frame = sum(self.processing_time_per_frame) / len(
                    self.processing_time_per_frame
                )

                with self.fps_lock:
                    current_fps = self.current_pipeline_fps
                estimated_fps = (
                    1.0 / avg_time_per_frame if avg_time_per_frame > 0 else current_fps
                )

                estimated_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
                with self.fps_lock:
                    self.current_pipeline_fps = estimated_fps

            self.last_fps_update = current_time

    def update_parameters(self, parameters: dict):
        """Update parameters that will be used in the next pipeline call."""
        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.info("Parameter queue full, dropping parameter update")
            return False

    def worker_loop(self):
        logger.info("WorkerFrameProcessor worker thread started")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(f"Error in worker loop: {e}")
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop: {e}, stopping frame processor"
                    )
                    self.stop()
                    break
        logger.info("WorkerFrameProcessor worker thread stopped")

    def process_chunk(self):
        start_time = time.time()
        try:
            # Check if there are new parameters
            try:
                new_parameters = self.parameters_queue.get_nowait()
                if new_parameters != self.parameters:
                    if (
                        "prompts" in new_parameters
                        and "transition" not in new_parameters
                        and "transition" in self.parameters
                    ):
                        self.parameters.pop("transition", None)

                    self.parameters = {**self.parameters, **new_parameters}
            except queue.Empty:
                pass

            # Pause or resume the processing
            paused = self.parameters.pop("paused", None)
            if paused is not None and paused != self.paused:
                self.paused = paused
            if self.paused:
                self.shutdown_event.wait(SLEEP_TIME)
                return

            reset_cache = self.parameters.pop("reset_cache", None)
            lora_scales = self.parameters.pop("lora_scales", None)

            if reset_cache:
                logger.info("Clearing output buffer queue due to reset_cache request")
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

            requirements = None
            if hasattr(self.pipeline, "prepare"):
                requirements = self.pipeline.prepare(**self.parameters)

            video_input = None
            if requirements is not None:
                current_chunk_size = requirements.input_size
                with self.frame_buffer_lock:
                    if not self.frame_buffer or len(self.frame_buffer) < current_chunk_size:
                        self.shutdown_event.wait(SLEEP_TIME)
                        return
                    video_input = self.prepare_chunk(current_chunk_size)

            call_params = dict(self.parameters.items())
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            if video_input is not None:
                call_params["video"] = video_input

            # Call pipeline directly - no proxy needed!
            output = self.pipeline(**call_params)

            # Clear transition when complete
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(self.pipeline, "state"):
                    transition_active = self.pipeline.state.get("_transition_active", False)

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    self.parameters.pop("transition", None)

            processing_time = time.time() - start_time
            num_frames = output.shape[0]
            logger.debug(
                f"Processed pipeline in {processing_time:.4f}s, {num_frames} frames"
            )

            # Normalize to [0, 255] and convert to uint8
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
                .cpu()
            )

            # Resize output queue to meet target max size
            target_output_queue_max_size = num_frames * OUTPUT_QUEUE_MAX_SIZE_FACTOR
            if self.output_queue.maxsize < target_output_queue_max_size:
                logger.info(
                    f"Increasing output queue size to {target_output_queue_max_size}, current size {self.output_queue.maxsize}, num_frames {num_frames}"
                )

                old_queue = self.output_queue
                self.output_queue = queue.Queue(maxsize=target_output_queue_max_size)
                while not old_queue.empty():
                    try:
                        frame = old_queue.get_nowait()
                        self.output_queue.put_nowait(frame)
                    except queue.Empty:
                        break

            for frame in output:
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    logger.warning("Output queue full, dropping processed frame")
                    self._calculate_pipeline_fps(start_time, num_frames)
                    continue

            self._calculate_pipeline_fps(start_time, num_frames)
        except Exception as e:
            if self._is_recoverable(e):
                logger.error(f"Error processing chunk: {e}", exc_info=True)
            else:
                raise e

        self.is_prepared = True

    def prepare_chunk(self, chunk_size: int) -> list[torch.Tensor]:
        """Sample frames uniformly from the buffer and convert them to tensors."""
        step = len(self.frame_buffer) / chunk_size
        indices = [round(i * step) for i in range(chunk_size)]
        video_frames_data = [self.frame_buffer[i] for i in indices]

        last_idx = indices[-1]
        for _ in range(last_idx + 1):
            self.frame_buffer.popleft()

        tensor_frames = []
        for frame_data in video_frames_data:
            # Convert frame data to tensor
            frame_array = frame_data.get("array")
            if frame_array is not None:
                tensor = torch.from_numpy(frame_array).float().unsqueeze(0)
                tensor_frames.append(tensor)

        return tensor_frames

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        return True


def _get_model_file_path(relative_path: str) -> str:
    """Get the absolute path to a model file."""
    models_dir = os.environ.get("MODELS_DIR", os.path.expanduser("~/.daydream-scope/models"))
    return os.path.join(models_dir, relative_path)


def _get_models_dir() -> str:
    """Get the models directory."""
    return os.environ.get("MODELS_DIR", os.path.expanduser("~/.daydream-scope/models"))


def _get_plugins_dir() -> str:
    """Get the plugins directory."""
    models_dir = _get_models_dir()
    # Get parent directory of models and add plugins
    parent_dir = os.path.dirname(models_dir)
    return os.path.join(parent_dir, "plugins")


def _load_plugin_pipeline(pipeline_id: str, load_params: dict | None = None):
    """Try to load a plugin pipeline from the plugins directory or current working directory.

    The worker is started with cwd set to the plugin directory, so we can import
    the pipeline module directly.
    """
    import importlib.util

    # First, try to find a pipeline.py in the current working directory
    cwd = os.getcwd()
    pipeline_module_path = os.path.join(cwd, "pipeline.py")

    if not os.path.exists(pipeline_module_path):
        # Try src/<plugin_id>/pipeline.py pattern
        src_dir = os.path.join(cwd, "src")
        if os.path.exists(src_dir):
            for name in os.listdir(src_dir):
                candidate = os.path.join(src_dir, name, "pipeline.py")
                if os.path.exists(candidate):
                    pipeline_module_path = candidate
                    break

    if not os.path.exists(pipeline_module_path):
        raise ValueError(f"No pipeline.py found for plugin pipeline: {pipeline_id}")

    logger.info(f"Loading plugin pipeline from: {pipeline_module_path}")

    # Dynamically import the pipeline module
    spec = importlib.util.spec_from_file_location("plugin_pipeline", pipeline_module_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load pipeline module from: {pipeline_module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["plugin_pipeline"] = module
    spec.loader.exec_module(module)

    # Look for a Pipeline class in the module
    pipeline_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and name.endswith("Pipeline") and name != "Pipeline":
            pipeline_class = obj
            break

    if pipeline_class is None:
        raise ValueError(f"No Pipeline class found in {pipeline_module_path}")

    logger.info(f"Found pipeline class: {pipeline_class.__name__}")

    # Instantiate the pipeline with load_params
    load_params = load_params or {}
    pipeline = pipeline_class(**load_params)

    logger.info(f"Plugin pipeline '{pipeline_id}' initialized in worker process")
    return pipeline


def _load_pipeline_implementation(pipeline_id: str, load_params: dict | None = None):
    """Load a pipeline in the worker process."""
    # Add the main project src to path so we can import scope modules
    project_root = os.environ.get("SCOPE_PROJECT_ROOT", "")
    if project_root:
        src_path = os.path.join(project_root, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    # List of built-in pipelines
    BUILTIN_PIPELINES = {
        "streamdiffusionv2",
        "passthrough",
        "longlive",
        "krea-realtime-video",
        "reward-forcing",
    }

    # If not a built-in pipeline, try to load as plugin
    if pipeline_id not in BUILTIN_PIPELINES:
        return _load_plugin_pipeline(pipeline_id, load_params)

    if pipeline_id == "streamdiffusionv2":
        from scope.core.pipelines import StreamDiffusionV2Pipeline

        models_dir = _get_models_dir()
        config = OmegaConf.create(
            {
                "model_dir": str(models_dir),
                "generator_path": str(
                    _get_model_file_path(
                        "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt"
                    )
                ),
                "text_encoder_path": str(
                    _get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
            }
        )

        # Apply load parameters (resolution, seed, LoRAs) to config
        height = 512
        width = 512
        seed = 42
        loras = None
        lora_merge_mode = "permanent_merge"

        if load_params:
            height = load_params.get("height", 512)
            width = load_params.get("width", 512)
            seed = load_params.get("seed", 42)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        config["_lora_merge_mode"] = lora_merge_mode

        pipeline = StreamDiffusionV2Pipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        logger.info("StreamDiffusionV2 pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "passthrough":
        from scope.core.pipelines import PassthroughPipeline

        # Use load parameters for resolution, default to 512x512
        height = 512
        width = 512
        if load_params:
            height = load_params.get("height", 512)
            width = load_params.get("width", 512)

        pipeline = PassthroughPipeline(
            height=height,
            width=width,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("Passthrough pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "longlive":
        from scope.core.pipelines import LongLivePipeline

        config = OmegaConf.create(
            {
                "model_dir": str(_get_models_dir()),
                "generator_path": str(
                    _get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
                ),
                "lora_path": str(
                    _get_model_file_path("LongLive-1.3B/models/lora.pt")
                ),
                "text_encoder_path": str(
                    _get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
            }
        )

        # Apply load parameters (resolution, seed, LoRAs) to config
        height = 320
        width = 576
        seed = 42
        loras = None
        lora_merge_mode = "permanent_merge"

        if load_params:
            height = load_params.get("height", 320)
            width = load_params.get("width", 576)
            seed = load_params.get("seed", 42)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        config["_lora_merge_mode"] = lora_merge_mode

        pipeline = LongLivePipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        logger.info("LongLive pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "krea-realtime-video":
        from scope.core.pipelines import KreaRealtimeVideoPipeline

        config = OmegaConf.create(
            {
                "model_dir": str(_get_models_dir()),
                "generator_path": str(
                    _get_model_file_path(
                        "krea-realtime-video/krea-realtime-video-14b.safetensors"
                    )
                ),
                "text_encoder_path": str(
                    _get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
                "vae_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                ),
            }
        )

        # Apply load parameters (resolution, seed, LoRAs) to config
        height = 512
        width = 512
        seed = 42
        loras = None
        lora_merge_mode = "permanent_merge"
        quantization = None

        if load_params:
            height = load_params.get("height", 512)
            width = load_params.get("width", 512)
            seed = load_params.get("seed", 42)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)
            quantization = load_params.get("quantization", None)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        config["_lora_merge_mode"] = lora_merge_mode

        pipeline = KreaRealtimeVideoPipeline(
            config,
            quantization=quantization,
            # Only compile diffusion model for hopper right now
            compile=any(
                x in torch.cuda.get_device_name(0).lower()
                for x in ("h100", "hopper")
            ),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("krea-realtime-video pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "reward-forcing":
        from scope.core.pipelines import RewardForcingPipeline

        config = OmegaConf.create(
            {
                "model_dir": str(_get_models_dir()),
                "generator_path": str(
                    _get_model_file_path("Reward-Forcing-T2V-1.3B/rewardforcing.pt")
                ),
                "text_encoder_path": str(
                    _get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
                "vae_path": str(
                    _get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
                ),
            }
        )

        # Apply load parameters (resolution, seed, LoRAs) to config
        height = 320
        width = 576
        seed = 42
        loras = None
        lora_merge_mode = "permanent_merge"
        quantization = None

        if load_params:
            height = load_params.get("height", 320)
            width = load_params.get("width", 576)
            seed = load_params.get("seed", 42)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)
            quantization = load_params.get("quantization", None)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        config["_lora_merge_mode"] = lora_merge_mode

        pipeline = RewardForcingPipeline(
            config,
            quantization=quantization,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("RewardForcing pipeline initialized in worker process")
        return pipeline

    else:
        # Fallback: try to load as plugin pipeline
        return _load_plugin_pipeline(pipeline_id, load_params)


def serialize_for_zmq(obj):
    """Serialize an object for ZMQ transmission."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tobytes(), "dtype": str(obj.dtype), "shape": obj.shape}
    elif isinstance(obj, torch.Tensor):
        arr = obj.cpu().numpy()
        return {"__ndarray__": True, "data": arr.tobytes(), "dtype": str(arr.dtype), "shape": arr.shape}
    elif isinstance(obj, dict):
        return {k: serialize_for_zmq(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_zmq(item) for item in obj]
    elif isinstance(obj, tuple):
        return {"__tuple__": True, "items": [serialize_for_zmq(item) for item in obj]}
    else:
        return obj


def deserialize_from_zmq(obj):
    """Deserialize an object from ZMQ transmission."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        elif obj.get("__tuple__"):
            return tuple(deserialize_from_zmq(item) for item in obj["items"])
        return {k: deserialize_from_zmq(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_zmq(item) for item in obj]
    return obj


def worker_main(command_port: int, response_port: int):
    """Main worker process function that handles pipeline and FrameProcessor operations via ZMQ.

    Args:
        command_port: ZMQ port for receiving commands
        response_port: ZMQ port for sending responses
    """
    # Set up logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [Worker-{os.getpid()}] - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Pipeline worker process started (PID: {os.getpid()})")
    logger.info(f"Connecting to command port {command_port}, response port {response_port}")

    # Set up ZMQ context and sockets
    context = zmq.Context()

    # PULL socket for receiving commands
    command_socket = context.socket(zmq.PULL)
    command_socket.connect(f"tcp://127.0.0.1:{command_port}")

    # PUSH socket for sending responses
    response_socket = context.socket(zmq.PUSH)
    response_socket.connect(f"tcp://127.0.0.1:{response_port}")

    pipeline = None
    pipeline_id = None
    frame_processors: dict[str, WorkerFrameProcessor] = {}

    try:
        while True:
            try:
                # Wait for commands from main process
                # Use a timeout so we can check for shutdown signals
                if command_socket.poll(timeout=100):
                    message = command_socket.recv()
                    command_data = json.loads(message.decode("utf-8"))

                    # Handle binary data separately if present
                    if command_data.get("__has_binary__"):
                        binary_data = command_socket.recv()
                        # Reconstruct the frame data
                        frame_info = command_data.get("frame_info", {})
                        if frame_info:
                            frame_array = np.frombuffer(
                                binary_data,
                                dtype=frame_info["dtype"]
                            ).reshape(frame_info["shape"])
                            command_data["frame_data"] = {"array": frame_array}
                else:
                    continue

                if command_data is None:
                    logger.info("Received shutdown signal")
                    break

                command = command_data.get("command")

                if command == WorkerCommand.LOAD_PIPELINE.value:
                    # Load pipeline
                    pipeline_id_to_load = command_data.get("pipeline_id")
                    load_params = command_data.get("load_params")

                    logger.info(
                        f"Loading pipeline: {pipeline_id_to_load} with params: {load_params}"
                    )

                    try:
                        # Unload existing pipeline if any
                        if pipeline is not None:
                            logger.info(f"Unloading existing pipeline: {pipeline_id}")
                            # Stop all frame processors
                            for fp_id, fp in list(frame_processors.items()):
                                fp.stop()
                                del frame_processors[fp_id]
                            del pipeline
                            pipeline = None
                            pipeline_id = None

                        # Load new pipeline
                        pipeline = _load_pipeline_implementation(
                            pipeline_id_to_load, load_params
                        )
                        pipeline_id = pipeline_id_to_load

                        response = {
                            "status": WorkerResponse.SUCCESS.value,
                            "message": f"Pipeline {pipeline_id} loaded successfully",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        logger.info(f"Pipeline {pipeline_id} loaded successfully")

                    except Exception as e:
                        error_msg = (
                            f"Failed to load pipeline: {str(e)}\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.CREATE_FRAME_PROCESSOR.value:
                    # Create a new FrameProcessor instance
                    if pipeline is None:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": "Pipeline not loaded",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        continue

                    try:
                        fp_id = command_data.get("frame_processor_id")
                        initial_parameters = command_data.get("initial_parameters", {})

                        if fp_id in frame_processors:
                            logger.warning(f"FrameProcessor {fp_id} already exists, stopping old one")
                            frame_processors[fp_id].stop()

                        frame_processor = WorkerFrameProcessor(
                            pipeline=pipeline,
                            initial_parameters=initial_parameters,
                        )
                        frame_processor.start()
                        frame_processors[fp_id] = frame_processor

                        response = {
                            "status": WorkerResponse.FRAME_PROCESSOR_CREATED.value,
                            "frame_processor_id": fp_id,
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        logger.info(f"Created FrameProcessor {fp_id}")

                    except Exception as e:
                        error_msg = (
                            f"Failed to create FrameProcessor: {str(e)}\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.DESTROY_FRAME_PROCESSOR.value:
                    # Destroy a FrameProcessor instance
                    fp_id = command_data.get("frame_processor_id")
                    if fp_id in frame_processors:
                        frame_processors[fp_id].stop()
                        del frame_processors[fp_id]
                        response = {
                            "status": WorkerResponse.SUCCESS.value,
                            "message": f"FrameProcessor {fp_id} destroyed",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        logger.info(f"Destroyed FrameProcessor {fp_id}")
                    else:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": f"FrameProcessor {fp_id} not found",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.PUT_FRAME.value:
                    # Put a frame into a FrameProcessor
                    fp_id = command_data.get("frame_processor_id")
                    frame_data = command_data.get("frame_data")

                    if fp_id not in frame_processors:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": f"FrameProcessor {fp_id} not found",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        continue

                    try:
                        success = frame_processors[fp_id].put(frame_data)
                        # Don't send response for every frame to avoid queue buildup
                        # Only send response if there's an error
                        if not success:
                            response = {
                                "status": WorkerResponse.ERROR.value,
                                "error": "Failed to put frame",
                            }
                            response_socket.send(json.dumps(response).encode("utf-8"))
                    except Exception as e:
                        error_msg = f"Error putting frame: {str(e)}"
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.GET_FRAME.value:
                    # Get a processed frame from a FrameProcessor
                    fp_id = command_data.get("frame_processor_id")

                    if fp_id not in frame_processors:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": f"FrameProcessor {fp_id} not found",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        continue

                    try:
                        frame_data = frame_processors[fp_id].get()
                        if frame_data is not None:
                            # Send frame data as binary for efficiency
                            arr = frame_data["data"]
                            header = {
                                "status": WorkerResponse.FRAME.value,
                                "__has_binary__": True,
                                "frame_info": {
                                    "dtype": str(arr.dtype),
                                    "shape": arr.shape,
                                },
                            }
                            response_socket.send(json.dumps(header).encode("utf-8"), zmq.SNDMORE)
                            response_socket.send(arr.tobytes())
                        else:
                            # No frame available - send empty response
                            response = {
                                "status": WorkerResponse.RESULT.value,
                                "result": None,
                            }
                            response_socket.send(json.dumps(response).encode("utf-8"))
                    except Exception as e:
                        error_msg = f"Error getting frame: {str(e)}"
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.UPDATE_PARAMETERS.value:
                    # Update parameters for a FrameProcessor
                    fp_id = command_data.get("frame_processor_id")
                    parameters = command_data.get("parameters", {})

                    if fp_id not in frame_processors:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": f"FrameProcessor {fp_id} not found",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        continue

                    try:
                        frame_processors[fp_id].update_parameters(parameters)
                        # Don't send response for parameter updates to avoid queue buildup
                    except Exception as e:
                        error_msg = f"Error updating parameters: {str(e)}"
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.GET_FPS.value:
                    # Get current FPS from a FrameProcessor
                    fp_id = command_data.get("frame_processor_id")

                    if fp_id not in frame_processors:
                        response = {
                            "status": WorkerResponse.ERROR.value,
                            "error": f"FrameProcessor {fp_id} not found",
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                        continue

                    try:
                        fps = frame_processors[fp_id].get_current_pipeline_fps()
                        response = {
                            "status": WorkerResponse.RESULT.value,
                            "result": fps,
                        }
                        response_socket.send(json.dumps(response).encode("utf-8"))
                    except Exception as e:
                        error_msg = f"Error getting FPS: {str(e)}"
                        logger.error(error_msg)
                        response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        response_socket.send(json.dumps(response).encode("utf-8"))

                elif command == WorkerCommand.SHUTDOWN.value:
                    logger.info("Received shutdown command")
                    break

            except zmq.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    # No message available, continue polling
                    continue
                else:
                    error_msg = f"ZMQ error: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                    response_socket.send(json.dumps(response).encode("utf-8"))

            except Exception as e:
                error_msg = (
                    f"Error processing command: {str(e)}\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                response = {"status": WorkerResponse.ERROR.value, "error": error_msg}
                response_socket.send(json.dumps(response).encode("utf-8"))

    finally:
        # Cleanup on exit
        logger.info("Cleaning up worker process...")
        # Stop all frame processors
        for fp_id, fp in list(frame_processors.items()):
            fp.stop()
        frame_processors.clear()
        if pipeline is not None:
            del pipeline

        # Close ZMQ sockets
        command_socket.close()
        response_socket.close()
        context.term()

        logger.info("Pipeline worker process shutting down")


def main():
    """Entry point for the worker process."""
    parser = argparse.ArgumentParser(description="Scope Pipeline Worker")
    parser.add_argument("--command-port", type=int, required=True, help="ZMQ command port")
    parser.add_argument("--response-port", type=int, required=True, help="ZMQ response port")

    args = parser.parse_args()

    worker_main(args.command_port, args.response_port)


if __name__ == "__main__":
    main()
