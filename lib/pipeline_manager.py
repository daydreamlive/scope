"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import gc
import logging
import os
import threading
from enum import Enum
from typing import Any

import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class PipelineNotAvailableException(Exception):
    """Exception raised when pipeline is not available for processing."""

    pass


class PipelineStatus(Enum):
    """Pipeline loading status enumeration."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class PipelineManager:
    """Manager for ML pipeline lifecycle."""

    def __init__(self):
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._model_config = None  # Store model config from model.yaml
        self._error_message = None
        self._lock = threading.RLock()  # Single reentrant lock for all access

    @property
    def status(self) -> PipelineStatus:
        """Get current pipeline status."""
        return self._status

    @property
    def pipeline_id(self) -> str | None:
        """Get current pipeline ID."""
        return self._pipeline_id

    @property
    def error_message(self) -> str | None:
        """Get last error message."""
        return self._error_message

    def get_pipeline(self):
        """Get the loaded pipeline instance (thread-safe)."""
        with self._lock:
            if self._status != PipelineStatus.LOADED or self._pipeline is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )
            return self._pipeline

    def get_status_info(self) -> dict[str, Any]:
        """Get detailed status information (thread-safe).

        Note: If status is ERROR, the error message is returned once and then cleared
        to prevent persistence across page reloads.
        """
        with self._lock:
            # Capture current state before clearing
            current_status = self._status
            error_message = self._error_message
            pipeline_id = self._pipeline_id
            load_params = self._load_params

            # If there's an error, clear it after capturing it
            # This ensures errors don't persist across page reloads
            if self._status == PipelineStatus.ERROR and error_message:
                self._error_message = None
                # Reset status to NOT_LOADED after error is retrieved
                self._status = PipelineStatus.NOT_LOADED
                self._pipeline_id = None
                self._load_params = None

            # Return the captured state (with error status if it was an error)
            return {
                "status": current_status.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
                "error": error_message,
            }

    async def get_pipeline_async(self):
        """Get the loaded pipeline instance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_pipeline)

    async def get_status_info_async(self) -> dict[str, Any]:
        """Get detailed status information (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_status_info)

    async def load_pipeline(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """
        Load a pipeline asynchronously.

        Args:
            pipeline_id: ID of pipeline to load. If None, uses PIPELINE env var.
            load_params: Pipeline-specific load parameters.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_pipeline_sync_wrapper, pipeline_id, load_params
        )

    def _load_pipeline_sync_wrapper(
        self, pipeline_id: str | None = None, load_params: dict | None = None
    ) -> bool:
        """Synchronous wrapper for pipeline loading with proper locking."""
        with self._lock:
            # If already loaded with same type and same params, return success
            # Normalize None to empty dict for comparison
            current_params = self._load_params or {}
            new_params = load_params or {}

            if (
                self._status == PipelineStatus.LOADED
                and self._pipeline_id == pipeline_id
                and current_params == new_params
            ):
                logger.info(
                    f"Pipeline {pipeline_id} already loaded with matching parameters"
                )
                return True

            # If a different pipeline is loaded OR same pipeline with different params, unload it first
            if self._status == PipelineStatus.LOADED and (
                self._pipeline_id != pipeline_id or current_params != new_params
            ):
                self._unload_pipeline_unsafe()

            # If already loading, someone else is handling it
            if self._status == PipelineStatus.LOADING:
                logger.info("Pipeline already loading by another thread")
                return False

            try:
                self._status = PipelineStatus.LOADING
                self._error_message = None

                # Determine pipeline type
                if pipeline_id is None:
                    pipeline_id = os.getenv("PIPELINE", "longlive")

                logger.info(f"Loading pipeline: {pipeline_id}")

                # Load the pipeline synchronously (we're already in executor thread)
                pipeline, model_config = self._load_pipeline_implementation(
                    pipeline_id, load_params
                )

                self._pipeline = pipeline
                self._pipeline_id = pipeline_id
                self._load_params = load_params
                self._model_config = model_config
                self._status = PipelineStatus.LOADED

                logger.info(f"Pipeline {pipeline_id} loaded successfully")
                return True

            except Exception as e:
                error_msg = f"Failed to load pipeline {pipeline_id}: {str(e)}"
                logger.error(error_msg)

                self._status = PipelineStatus.ERROR
                self._error_message = error_msg
                self._pipeline = None
                self._pipeline_id = None
                self._load_params = None
                self._model_config = None

                return False

    def _unload_pipeline_unsafe(self):
        """Unload the current pipeline. Must be called with lock held."""
        if self._pipeline:
            logger.info(f"Unloading pipeline: {self._pipeline_id}")

        # Change status and pipeline atomically
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
        self._model_config = None
        self._error_message = None

        # Cleanup resources
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("CUDA cache cleared")
            except Exception as e:
                logger.warning(f"CUDA cleanup failed: {e}")

    def _load_pipeline_implementation(
        self, pipeline_id: str, load_params: dict | None = None
    ):
        """Synchronous pipeline loading (runs in thread executor).

        Returns:
            tuple: (pipeline_instance, model_config) where model_config is the loaded
                   model.yaml config or None if pipeline has no model.yaml
        """
        if pipeline_id == "streamdiffusionv2":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

            models_dir = get_models_dir()
            model_config = OmegaConf.load("pipelines/streamdiffusionv2/model.yaml")
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
                    "model_config": model_config,
                }
            )

            # Use load parameters for resolution and seed
            height = 512
            width = 512
            seed = 42
            if load_params:
                height = load_params.get("height", 512)
                width = load_params.get("width", 512)
                seed = load_params.get("seed", 42)

            config["height"] = height
            config["width"] = width
            config["seed"] = seed

            pipeline = StreamDiffusionV2Pipeline(
                config, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            logger.info("StreamDiffusionV2 pipeline initialized")
            return pipeline, model_config

        elif pipeline_id == "passthrough":
            from pipelines.passthrough.pipeline import PassthroughPipeline

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
            logger.info("Passthrough pipeline initialized")
            return pipeline, None  # No model.yaml for passthrough

        elif pipeline_id == "vod":
            from pipelines.vod.pipeline import VodPipeline

            # Use load parameters for resolution, default to 512x512
            height = 512
            width = 512
            if load_params:
                height = load_params.get("height", 512)
                width = load_params.get("width", 512)

            pipeline = VodPipeline(
                height=height,
                width=width,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("VOD pipeline initialized")
            return pipeline, None  # No model.yaml for vod

        elif pipeline_id == "longlive":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.longlive.pipeline import LongLivePipeline

            model_config = OmegaConf.load("pipelines/longlive/model.yaml")
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
                    "model_config": model_config,
                }
            )

            height = 320
            width = 576
            seed = 42
            if load_params:
                height = load_params.get("height", 320)
                width = load_params.get("width", 576)
                seed = load_params.get("seed", 42)

            config["height"] = height
            config["width"] = width
            config["seed"] = seed

            pipeline = LongLivePipeline(
                config, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            logger.info("LongLive pipeline initialized")
            return pipeline, model_config

        elif pipeline_id == "krea-realtime-video":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

            model_config = OmegaConf.load("pipelines/krea_realtime_video/model.yaml")
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
                    "model_config": model_config,
                }
            )

            height = 512
            width = 512
            seed = 42
            quantization = None
            if load_params:
                height = load_params.get("height", 512)
                width = load_params.get("width", 512)
                seed = load_params.get("seed", 42)
                quantization = load_params.get("quantization", None)

            config["height"] = height
            config["width"] = width
            config["seed"] = seed

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
            logger.info("krea-realtime-video pipeline initialized")
            return pipeline, model_config

        else:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")

    def unload_pipeline(self):
        """Unload the current pipeline (thread-safe)."""
        with self._lock:
            self._unload_pipeline_unsafe()

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED

    def get_required_input_frames(self, pipeline_id: str | None = None) -> int:
        """
        Calculate the required number of input frames for a pipeline.

        Args:
            pipeline_id: ID of pipeline. If None, uses current loaded pipeline ID.

        Returns:
            int: Required number of input frames:
                - 1 if pipeline has no model.yaml
                - num_frame_per_block * vae_temporal_downsample_factor if both exist in model.yaml
                - 0 otherwise (should be treated as "now" - no buffering required)
        """
        with self._lock:
            # If no model config is stored, pipeline has no model.yaml
            if self._model_config is None:
                return 1

            try:
                # Extract fields from stored model config
                num_frame_per_block = self._model_config.get("num_frame_per_block")
                vae_temporal_downsample_factor = self._model_config.get(
                    "vae_temporal_downsample_factor"
                )

                if (
                    num_frame_per_block is not None
                    and vae_temporal_downsample_factor is not None
                ):
                    return num_frame_per_block * vae_temporal_downsample_factor
                else:
                    # Otherwise, return 0 (should be treated as "now")
                    return 0
            except Exception as e:
                logger.warning(f"Error reading model config: {e}. Returning 0.")
                return 0
