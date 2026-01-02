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


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, CPU otherwise)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self._depth_preprocessor = None  # Video-Depth-Anything preprocessor (sync)
        self._async_depth_preprocessor = None  # Async depth preprocessor client

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

    @property
    def depth_preprocessor(self):
        """Get the loaded sync depth preprocessor (if any)."""
        return self._depth_preprocessor

    @property
    def async_depth_preprocessor(self):
        """Get the async depth preprocessor client (if any)."""
        return self._async_depth_preprocessor

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

        if pipeline_id is None:
            pipeline_id = os.getenv("PIPELINE", "longlive")

        with self._lock:
            # Normalize None to empty dict for comparison
            current_params = self._load_params or {}
            new_params = load_params or {}

            # If already loaded with same type and same params, return success
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

            # Mark as loading
            self._status = PipelineStatus.LOADING
            self._error_message = None

        # Release lock during slow loading operation
        logger.info(f"Loading pipeline: {pipeline_id}")

        try:
            # Load the pipeline synchronously (we're already in executor thread)
            pipeline = self._load_pipeline_implementation(pipeline_id, load_params)

            # Hold lock while updating state with loaded pipeline
            with self._lock:
                self._pipeline = pipeline
                self._pipeline_id = pipeline_id
                self._load_params = load_params
                self._status = PipelineStatus.LOADED

            logger.info(f"Pipeline {pipeline_id} loaded successfully")
            return True

        except Exception as e:
            from .models_config import get_models_dir

            models_dir = get_models_dir()
            error_msg = f"Failed to load pipeline {pipeline_id}: {e}"
            logger.error(
                f"{error_msg}. If this error persists, consider removing the models "
                f"directory '{models_dir}' and re-downloading models."
            )

            # Hold lock while updating state with error
            with self._lock:
                self._status = PipelineStatus.ERROR
                self._error_message = error_msg
                self._pipeline = None
                self._pipeline_id = None
                self._load_params = None

            return False

    def _get_vace_checkpoint_path(self) -> str:
        """Get the path to the VACE module checkpoint.

        Returns:
            str: Path to VACE module checkpoint file (contains only VACE weights)
        """
        from .models_config import get_model_file_path

        return str(
            get_model_file_path(
                "WanVideo_comfy/Wan2_1-VACE_module_1_3B_bf16.safetensors"
            )
        )

    def _configure_vace(self, config: dict, load_params: dict | None = None) -> None:
        """Configure VACE support for a pipeline.

        Adds vace_path to config and optionally extracts VACE-specific parameters
        from load_params (ref_images, vace_context_scale).

        Args:
            config: Pipeline configuration dict to modify
            load_params: Optional load parameters containing VACE settings
        """
        config["vace_path"] = self._get_vace_checkpoint_path()
        logger.debug(f"_configure_vace: Using VACE checkpoint at {config['vace_path']}")

        # Extract VACE-specific parameters from load_params if present
        if load_params:
            ref_images = load_params.get("ref_images", [])
            if ref_images:
                config["ref_images"] = ref_images
                config["vace_context_scale"] = load_params.get(
                    "vace_context_scale", 1.0
                )
                logger.info(
                    f"_configure_vace: VACE parameters from load_params: "
                    f"ref_images count={len(ref_images)}, "
                    f"vace_context_scale={config.get('vace_context_scale', 1.0)}"
                )

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

        # Unload async depth preprocessor if loaded
        if self._async_depth_preprocessor is not None:
            logger.info("Stopping async depth preprocessor")
            self._async_depth_preprocessor.stop()
            self._async_depth_preprocessor = None

        # Unload sync depth preprocessor if loaded
        if self._depth_preprocessor is not None:
            logger.info("Unloading sync depth preprocessor")
            self._depth_preprocessor.offload()
            self._depth_preprocessor = None

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

    def _load_depth_preprocessor(self, encoder: str, use_async: bool = True) -> None:
        """Load the Video-Depth-Anything preprocessor.

        Args:
            encoder: Encoder size ("vits", "vitb", or "vitl")
            use_async: If True, use async preprocessor (separate process with ZeroMQ)
        """
        if use_async:
            # Load async depth preprocessor (runs in separate process)
            from scope.core.preprocessors import DepthPreprocessorClient

            logger.info(
                f"Loading async Video-Depth-Anything preprocessor with encoder: {encoder}"
            )
            self._async_depth_preprocessor = DepthPreprocessorClient(
                encoder=encoder,
            )
            if self._async_depth_preprocessor.start():
                logger.info("Async Video-Depth-Anything preprocessor started")
            else:
                logger.error("Failed to start async depth preprocessor, falling back to sync")
                self._async_depth_preprocessor = None
                # Fall back to sync preprocessor
                self._load_sync_depth_preprocessor(encoder)
        else:
            self._load_sync_depth_preprocessor(encoder)

    def _load_sync_depth_preprocessor(self, encoder: str) -> None:
        """Load the synchronous Video-Depth-Anything preprocessor.

        Args:
            encoder: Encoder size ("vits", "vitb", or "vitl")
        """
        from scope.core.preprocessors import VideoDepthAnything

        logger.info(f"Loading sync Video-Depth-Anything preprocessor with encoder: {encoder}")
        self._depth_preprocessor = VideoDepthAnything(
            encoder=encoder,
            device=torch.device("cuda"),
            dtype=torch.float16,  # VDA uses fp16
        )
        self._depth_preprocessor.load_model()
        logger.info("Sync Video-Depth-Anything preprocessor loaded")

    def _load_pipeline_implementation(
        self, pipeline_id: str, load_params: dict | None = None
    ):
        """Synchronous pipeline loading (runs in thread executor)."""
        from scope.core.pipelines.registry import PipelineRegistry

        # Check if pipeline is in registry
        pipeline_class = PipelineRegistry.get(pipeline_id)

        # List of built-in pipelines with custom initialization
        BUILTIN_PIPELINES = {
            "streamdiffusionv2",
            "passthrough",
            "depthanything",
            "longlive",
            "krea-realtime-video",
            "reward-forcing",
            "memflow",
        }

        if pipeline_class is not None and pipeline_id not in BUILTIN_PIPELINES:
            # Plugin pipeline - instantiate generically with load_params
            logger.info(f"Loading plugin pipeline: {pipeline_id}")
            load_params = load_params or {}
            return pipeline_class(**load_params)

        # Fall through to built-in pipeline initialization
        if pipeline_id == "streamdiffusionv2":
            from scope.core.pipelines import (
                StreamDiffusionV2Pipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

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

            # Configure VACE support if enabled in load_params (default: True)
            # Note: VACE is not available for StreamDiffusion in video mode (enforced by frontend)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)

            pipeline = StreamDiffusionV2Pipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("StreamDiffusionV2 pipeline initialized")

            # Load depth preprocessor if specified
            depth_encoder = None
            if load_params:
                depth_encoder = load_params.get("depth_preprocessor_encoder", None)
            if depth_encoder:
                self._load_depth_preprocessor(depth_encoder)

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
                device=get_device(),
                dtype=torch.bfloat16,
            )
            logger.info("Passthrough pipeline initialized")
            return pipeline

        elif pipeline_id == "depthanything":
            from scope.core.pipelines.depthanything import DepthAnythingPipeline

            # Use load parameters for resolution and encoder, with defaults
            height = 480
            width = 848
            encoder = "vitl"
            input_size = 392
            streaming = True
            output_format = "grayscale"
            if load_params:
                height = load_params.get("height", height)
                width = load_params.get("width", width)
                encoder = load_params.get("encoder", encoder)
                input_size = load_params.get("input_size", input_size)
                streaming = load_params.get("streaming", streaming)
                output_format = load_params.get("output_format", output_format)

            pipeline = DepthAnythingPipeline(
                height=height,
                width=width,
                encoder=encoder,
                device=get_device(),
                dtype=torch.float16,
                input_size=input_size,
                streaming=streaming,
                output_format=output_format,
            )
            logger.info("DepthAnything pipeline initialized")
            return pipeline

        elif pipeline_id == "longlive":
            from scope.core.pipelines import LongLivePipeline

            from .models_config import get_model_file_path, get_models_dir

            models_dir = get_models_dir()
            config = OmegaConf.create(
                {
                    "model_dir": str(models_dir),
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

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=320,
                default_width=576,
                default_seed=42,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)

            pipeline = LongLivePipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("LongLive pipeline initialized")

            # Load depth preprocessor if specified
            depth_encoder = None
            if load_params:
                depth_encoder = load_params.get("depth_preprocessor_encoder", None)
            if depth_encoder:
                self._load_depth_preprocessor(depth_encoder)

            return pipeline

        elif pipeline_id == "krea-realtime-video":
            from scope.core.pipelines import (
                KreaRealtimeVideoPipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

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
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
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

        elif pipeline_id == "reward-forcing":
            from scope.core.pipelines import (
                RewardForcingPipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

            config = OmegaConf.create(
                {
                    "model_dir": str(get_models_dir()),
                    "generator_path": str(
                        get_model_file_path("Reward-Forcing-T2V-1.3B/rewardforcing.pt")
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

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=320,
                default_width=576,
                default_seed=42,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)

            pipeline = RewardForcingPipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("RewardForcing pipeline initialized")
            return pipeline

        elif pipeline_id == "memflow":
            from scope.core.pipelines import (
                MemFlowPipeline,
            )

            from .models_config import get_model_file_path, get_models_dir

            models_dir = get_models_dir()
            config = OmegaConf.create(
                {
                    "model_dir": str(models_dir),
                    "generator_path": str(get_model_file_path("MemFlow/base.pt")),
                    "lora_path": str(get_model_file_path("MemFlow/lora.pt")),
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

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                self._configure_vace(config, load_params)
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

            # Apply load parameters (resolution, seed, LoRAs) to config
            self._apply_load_params(
                config,
                load_params,
                default_height=320,
                default_width=576,
                default_seed=42,
            )

            quantization = None
            if load_params:
                quantization = load_params.get("quantization", None)

            pipeline = MemFlowPipeline(
                config,
                quantization=quantization,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )
            logger.info("MemFlow pipeline initialized")
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
