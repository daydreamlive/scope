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
    HAS_ATTR = "has_attr"
    SHUTDOWN = "shutdown"


class WorkerResponse(Enum):
    """Response types from worker process."""

    SUCCESS = "success"
    ERROR = "error"
    PIPELINE_LOADED = "pipeline_loaded"
    PIPELINE_NOT_LOADED = "pipeline_not_loaded"
    RESULT = "result"


def _load_pipeline_implementation(pipeline_id: str, load_params: dict | None = None):
    """Load a pipeline in the worker process.

    This is the same logic as in PipelineManager._load_pipeline_implementation
    but runs in a separate process for proper VRAM isolation.
    """
    if pipeline_id == "streamdiffusionv2":
        from scope.core.pipelines import (
            StreamDiffusionV2Pipeline,
        )

        from scope.server.models_config import get_model_file_path, get_models_dir

        models_dir = get_models_dir()
        config = OmegaConf.create(
            {
                "model_dir": str(models_dir),
                "generator_path": str(
                    get_model_file_path(
                        "StreamDiffusionV2/wan_causal_dmd_v2v/model.pt"
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

        from scope.server.models_config import get_model_file_path, get_models_dir

        config = OmegaConf.create(
            {
                "model_dir": str(get_models_dir()),
                "generator_path": str(
                    get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
                ),
                "lora_path": str(
                    get_model_file_path("LongLive-1.3B/models/lora.pt")
                ),
                "text_encoder_path": str(
                    get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
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
        from scope.core.pipelines import (
            KreaRealtimeVideoPipeline,
        )

        from scope.server.models_config import get_model_file_path, get_models_dir

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
                "vae_path": str(
                    get_model_file_path("Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
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
                        error_msg = (
                            f"Failed to load pipeline: {str(e)}\n{traceback.format_exc()}"
                        )
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

                        # Check if method exists on pipeline
                        if not hasattr(pipeline, method):
                            raise AttributeError(
                                f"Pipeline does not have method '{method}'"
                            )

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

                    except AttributeError as e:
                        # Re-raise AttributeError so hasattr() checks work correctly
                        error_msg = f"AttributeError: {str(e)}"
                        logger.debug(error_msg)
                        response_queue.put(
                            {"status": WorkerResponse.ERROR.value, "error": error_msg}
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

                elif command == WorkerCommand.HAS_ATTR.value:
                    # Check if pipeline has an attribute/method
                    if pipeline is None:
                        response_queue.put(
                            {
                                "status": WorkerResponse.ERROR.value,
                                "error": "Pipeline not loaded",
                            }
                        )
                        continue

                    attr_name = command_data.get("attr_name")
                    has_attr = hasattr(pipeline, attr_name)
                    response_queue.put(
                        {
                            "status": WorkerResponse.RESULT.value,
                            "result": has_attr,
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
