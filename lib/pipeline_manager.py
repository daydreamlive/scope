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

            # Capture loaded LoRA adapters if pipeline exposes them
            loaded_lora_adapters = None
            if self._pipeline is not None and hasattr(
                self._pipeline, "loaded_lora_adapters"
            ):
                loaded_lora_adapters = getattr(
                    self._pipeline, "loaded_lora_adapters", None
                )

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
                "loaded_lora_adapters": loaded_lora_adapters,
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
                pipeline = self._load_pipeline_implementation(pipeline_id, load_params)

                self._pipeline = pipeline
                self._pipeline_id = pipeline_id
                self._load_params = load_params
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

                return False

    def _apply_load_params(
        self,
        config: dict,
        load_params: dict | None,
        default_height: int,
        default_width: int,
        default_seed: int = 42,
    ) -> None:
        """Extract and apply common load parameters (resolution, seed, LoRAs) to config.

        Args:
            config: Pipeline config dict to update
            load_params: Load parameters dict (may contain height, width, seed, loras, lora_merge_mode)
            default_height: Default height if not in load_params
            default_width: Default width if not in load_params
            default_seed: Default seed if not in load_params
        """
        height = default_height
        width = default_width
        seed = default_seed
        loras = None
        lora_merge_mode = "permanent_merge"

        if load_params:
            height = load_params.get("height", default_height)
            width = load_params.get("width", default_width)
            seed = load_params.get("seed", default_seed)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        if loras:
            config["loras"] = loras
        # Pass merge_mode directly to mixin, not via config
        config["_lora_merge_mode"] = lora_merge_mode

    def _unload_pipeline_unsafe(self):
        """Unload the current pipeline. Must be called with lock held."""
        if self._pipeline:
            logger.info(f"Unloading pipeline: {self._pipeline_id}")

        # Change status and pipeline atomically
        self._status = PipelineStatus.NOT_LOADED
        self._pipeline = None
        self._pipeline_id = None
        self._load_params = None
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
        """Synchronous pipeline loading (runs in thread executor)."""
        from lib.defaults import get_mode_defaults

        if pipeline_id == "streamdiffusionv2":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

            native_defaults = get_mode_defaults(StreamDiffusionV2Pipeline)
            default_resolution = native_defaults.get("resolution", {})
            default_height = default_resolution.get("height", 512)
            default_width = default_resolution.get("width", 512)
            default_seed = native_defaults.get("base_seed", 42)

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
                    "model_config": OmegaConf.load(
                        "pipelines/streamdiffusionv2/model.yaml"
                    ),
                }
            )

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=default_height,
                default_width=default_width,
                default_seed=default_seed,
            )

            pipeline = StreamDiffusionV2Pipeline(
                config, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            logger.info("StreamDiffusionV2 pipeline initialized")
            return pipeline

        elif pipeline_id == "passthrough":
            from pipelines.passthrough.pipeline import PassthroughPipeline

            native_defaults = get_mode_defaults(PassthroughPipeline)
            default_resolution = native_defaults.get("resolution", {})
            default_height = default_resolution.get("height", 512)
            default_width = default_resolution.get("width", 512)

            # Use load parameters for resolution, default to 512x512
            height = default_height
            width = default_width
            if load_params:
                height = load_params.get("height", default_height)
                width = load_params.get("width", default_width)

            pipeline = PassthroughPipeline(
                height=height,
                width=width,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("Passthrough pipeline initialized")
            return pipeline

        elif pipeline_id == "longlive":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.longlive.pipeline import LongLivePipeline

            native_defaults = get_mode_defaults(LongLivePipeline)
            default_resolution = native_defaults.get("resolution", {})
            default_height = default_resolution.get("height", 320)
            default_width = default_resolution.get("width", 576)
            default_seed = native_defaults.get("base_seed", 42)

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
                    "model_config": OmegaConf.load("pipelines/longlive/model.yaml"),
                }
            )

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=default_height,
                default_width=default_width,
                default_seed=default_seed,
            )

            pipeline = LongLivePipeline(
                config, device=torch.device("cuda"), dtype=torch.bfloat16
            )
            logger.info("LongLive pipeline initialized")
            return pipeline

        elif pipeline_id == "krea-realtime-video":
            from lib.models_config import get_model_file_path, get_models_dir
            from pipelines.krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

            native_defaults = get_mode_defaults(KreaRealtimeVideoPipeline)
            default_resolution = native_defaults.get("resolution", {})
            default_height = default_resolution.get("height", 320)
            default_width = default_resolution.get("width", 576)
            default_seed = native_defaults.get("base_seed", 42)

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
                    "model_config": OmegaConf.load(
                        "pipelines/krea_realtime_video/model.yaml"
                    ),
                }
            )

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=default_height,
                default_width=default_width,
                default_seed=default_seed,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)

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
            return pipeline

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
