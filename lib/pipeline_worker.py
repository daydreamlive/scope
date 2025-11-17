"""Pipeline Worker - Runs pipeline in a separate process for proper VRAM cleanup."""

import logging
import multiprocessing as mp
import os
import traceback
from enum import Enum

import torch
from omegaconf import OmegaConf

# Configure logging for worker process
logger = logging.getLogger(__name__)


class WorkerCommand(Enum):
    """Commands that can be sent to the worker process."""

    LOAD_PIPELINE = "load_pipeline"
    UNLOAD_PIPELINE = "unload_pipeline"
    GET_PIPELINE = "get_pipeline"
    CALL_PIPELINE = "call_pipeline"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    """Response types from worker process."""

    SUCCESS = "success"
    ERROR = "error"
    PIPELINE_LOADED = "pipeline_loaded"
    PIPELINE_NOT_LOADED = "pipeline_not_loaded"
    RESULT = "result"


def _extract_load_params(load_params: dict | None, defaults: dict) -> dict:
    """Extract load parameters with defaults.

    Args:
        load_params: User-provided load parameters
        defaults: Dictionary of default values

    Returns:
        Dictionary with extracted parameters
    """
    if load_params is None:
        return defaults.copy()
    return {key: load_params.get(key, default) for key, default in defaults.items()}


def _load_pipeline_implementation(pipeline_id: str, load_params: dict | None = None):
    """Load a pipeline in the worker process.

    This is the same logic as in PipelineManager._load_pipeline_implementation
    but runs in a separate process for proper VRAM isolation.
    """
    if pipeline_id == "streamdiffusionv2":
        from lib.models_config import get_model_file_path, get_models_dir
        from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        config = OmegaConf.load("pipelines/streamdiffusionv2/model.yaml")
        models_dir = get_models_dir()
        config["model_dir"] = str(models_dir)
        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )

        # Extract load parameters with defaults
        params = _extract_load_params(
            load_params, {"height": 512, "width": 512, "seed": 42}
        )
        config["height"] = params["height"]
        config["width"] = params["width"]
        config["seed"] = params["seed"]

        pipeline = StreamDiffusionV2Pipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        logger.info("StreamDiffusionV2 pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "passthrough":
        from pipelines.passthrough.pipeline import PassthroughPipeline

        # Extract load parameters with defaults
        params = _extract_load_params(load_params, {"height": 512, "width": 512})
        pipeline = PassthroughPipeline(
            height=params["height"],
            width=params["width"],
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("Passthrough pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "vod":
        from pipelines.vod.pipeline import VodPipeline

        # Extract load parameters with defaults
        params = _extract_load_params(load_params, {"height": 512, "width": 512})
        pipeline = VodPipeline(
            height=params["height"],
            width=params["width"],
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("VOD pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "longlive":
        from lib.models_config import get_model_file_path, get_models_dir
        from pipelines.longlive.pipeline import LongLivePipeline

        config = OmegaConf.load("pipelines/longlive/model.yaml")
        models_dir = get_models_dir()
        config["model_dir"] = str(models_dir)
        config["generator_path"] = get_model_file_path(
            "LongLive-1.3B/models/longlive_base.pt"
        )
        config["lora_path"] = get_model_file_path("LongLive-1.3B/models/lora.pt")
        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )

        # Extract load parameters with defaults
        params = _extract_load_params(
            load_params, {"height": 320, "width": 576, "seed": 42}
        )
        config["height"] = params["height"]
        config["width"] = params["width"]
        config["seed"] = params["seed"]

        pipeline = LongLivePipeline(
            config, device=torch.device("cuda"), dtype=torch.bfloat16
        )
        logger.info("LongLive pipeline initialized in worker process")
        return pipeline

    elif pipeline_id == "krea-realtime-video":
        from lib.models_config import get_model_file_path, get_models_dir
        from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

        config = OmegaConf.load("pipelines/krea_realtime_video/model.yaml")
        models_dir = get_models_dir()
        config["model_dir"] = str(models_dir)
        config["generator_path"] = str(
            get_model_file_path(
                "krea-realtime-video/krea-realtime-video-14b.safetensors"
            )
        )
        config["text_encoder_path"] = str(
            get_model_file_path("WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors")
        )
        config["tokenizer_path"] = str(
            get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
        )
        config["vae_path"] = str(get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))

        # Extract load parameters with defaults
        params = _extract_load_params(
            load_params, {"height": 512, "width": 512, "seed": 42, "quantization": None}
        )
        config["height"] = params["height"]
        config["width"] = params["width"]
        config["seed"] = params["seed"]
        quantization = params["quantization"]

        pipeline = KreaRealtimeVideoPipeline(
            config,
            quantization=quantization,
            # Only compile diffusion model for hopper right now
            compile=any(
                x in torch.cuda.get_device_name(0).lower() for x in ("h100", "hopper")
            ),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("krea-realtime-video pipeline initialized in worker process")
        return pipeline

    else:
        raise ValueError(f"Invalid pipeline ID: {pipeline_id}")


def pipeline_worker_process(command_queue: mp.Queue, response_queue: mp.Queue):
    """Main worker process function that handles pipeline operations.

    This process runs in isolation and can be killed to ensure proper VRAM cleanup.

    Args:
        command_queue: Queue for receiving commands from main process
        response_queue: Queue for sending responses back to main process
    """
    # Set up logging for worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - [Worker-{os.getpid()}] - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Pipeline worker process started (PID: {os.getpid()})")

    pipeline = None
    pipeline_id = None

    try:
        while True:
            try:
                # Wait for commands from main process
                command_data = command_queue.get()

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
                            del pipeline
                            pipeline = None
                            pipeline_id = None

                        # Load new pipeline
                        pipeline = _load_pipeline_implementation(
                            pipeline_id_to_load, load_params
                        )
                        pipeline_id = pipeline_id_to_load

                        response_queue.put(
                            {
                                "status": WorkerResponse.SUCCESS.value,
                                "message": f"Pipeline {pipeline_id} loaded successfully",
                            }
                        )
                        logger.info(f"Pipeline {pipeline_id} loaded successfully")

                    except Exception as e:
                        error_msg = f"Failed to load pipeline: {str(e)}\n{traceback.format_exc()}"
                        logger.error(error_msg)
                        response_queue.put(
                            {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        )

                elif command == WorkerCommand.CALL_PIPELINE.value:
                    # Call pipeline with provided arguments
                    if pipeline is None:
                        response_queue.put(
                            {
                                "status": WorkerResponse.ERROR.value,
                                "error": "Pipeline not loaded",
                            }
                        )
                        continue

                    try:
                        # Get method name and arguments
                        method = command_data.get("method", "__call__")
                        args = command_data.get("args", [])
                        kwargs = command_data.get("kwargs", {})

                        # Deserialize tensors if needed
                        args = _deserialize_tensors(args)
                        kwargs = _deserialize_tensors(kwargs)

                        # Call the pipeline method
                        pipeline_method = getattr(pipeline, method)
                        result = pipeline_method(*args, **kwargs)

                        # Serialize result for transmission
                        serialized_result = _serialize_tensors(result)

                        response_queue.put(
                            {
                                "status": WorkerResponse.RESULT.value,
                                "result": serialized_result,
                            }
                        )

                    except Exception as e:
                        error_msg = (
                            f"Pipeline call failed: {str(e)}\n{traceback.format_exc()}"
                        )
                        logger.error(error_msg)
                        response_queue.put(
                            {"status": WorkerResponse.ERROR.value, "error": error_msg}
                        )

                elif command == WorkerCommand.GET_PIPELINE.value:
                    # Check if pipeline is loaded
                    if pipeline is None:
                        response_queue.put(
                            {"status": WorkerResponse.PIPELINE_NOT_LOADED.value}
                        )
                    else:
                        response_queue.put(
                            {
                                "status": WorkerResponse.PIPELINE_LOADED.value,
                                "pipeline_id": pipeline_id,
                            }
                        )

                elif command == WorkerCommand.SHUTDOWN.value:
                    logger.info("Received shutdown command")
                    break

            except Exception as e:
                error_msg = (
                    f"Error processing command: {str(e)}\n{traceback.format_exc()}"
                )
                logger.error(error_msg)
                response_queue.put(
                    {"status": WorkerResponse.ERROR.value, "error": error_msg}
                )

    finally:
        # Cleanup on exit
        logger.info("Cleaning up worker process...")
        if pipeline is not None:
            del pipeline

        logger.info("Pipeline worker process shutting down")


def _serialize_tensors(obj):
    """Serialize torch tensors for inter-process communication.

    For CUDA tensors, we move them to CPU first for serialization.
    """
    if isinstance(obj, torch.Tensor):
        # Move to CPU for serialization
        return {"__tensor__": True, "data": obj.cpu().numpy()}
    elif isinstance(obj, dict):
        return {k: _serialize_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_tensors(item) for item in obj)
    else:
        return obj


def _deserialize_tensors(obj):
    """Deserialize torch tensors from inter-process communication."""
    if isinstance(obj, dict):
        if obj.get("__tensor__"):
            # Reconstruct tensor from numpy array
            return torch.from_numpy(obj["data"])
        return {k: _deserialize_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_tensors(item) for item in obj)
    return obj
