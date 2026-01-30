"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import gc
import logging
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

        # Support for multiple pipelines (for pipeline chaining)
        self._pipelines: dict[str, Any] = {}  # pipeline_id -> pipeline instance
        self._pipeline_statuses: dict[str, PipelineStatus] = {}  # pipeline_id -> status
        self._pipeline_load_params: dict[str, dict] = {}  # pipeline_id -> load_params

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
        """Get the loaded pipeline instance (thread-safe).

        This is the legacy method that returns the main pipeline.
        For pipeline chaining, use get_pipeline_by_id() instead.
        """
        with self._lock:
            if self._status != PipelineStatus.LOADED or self._pipeline is None:
                raise PipelineNotAvailableException(
                    f"Pipeline not available. Status: {self._status.value}"
                )
            return self._pipeline

    def get_pipeline_by_id(self, pipeline_id: str):
        """Get a pipeline instance by ID (thread-safe).

        Args:
            pipeline_id: ID of the pipeline to retrieve

        Returns:
            Pipeline instance

        Raises:
            PipelineNotAvailableException: If pipeline is not loaded
        """
        with self._lock:
            if pipeline_id not in self._pipelines:
                raise PipelineNotAvailableException(
                    f"Pipeline {pipeline_id} not loaded"
                )
            status = self._pipeline_statuses.get(pipeline_id, PipelineStatus.NOT_LOADED)
            if status != PipelineStatus.LOADED:
                raise PipelineNotAvailableException(
                    f"Pipeline {pipeline_id} not available. Status: {status.value}"
                )
            return self._pipelines[pipeline_id]

    async def _load_pipeline_by_id(
        self, pipeline_id: str, load_params: dict | None = None
    ) -> bool:
        """
        Load a pipeline by ID asynchronously (private method).

        Args:
            pipeline_id: ID of pipeline to load
            load_params: Pipeline-specific load parameters

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_pipeline_by_id_sync, pipeline_id, load_params
        )

    def _load_pipeline_by_id_sync(
        self, pipeline_id: str, load_params: dict | None = None
    ) -> bool:
        """Synchronous wrapper for loading a pipeline by ID."""
        with self._lock:
            # Check if already loaded with same params
            current_params = self._pipeline_load_params.get(pipeline_id, {})
            new_params = load_params or {}

            # Check if pipeline is already loaded (either in _pipelines or as main pipeline)
            is_loaded = False
            if pipeline_id in self._pipelines:
                if (
                    self._pipeline_statuses.get(pipeline_id) == PipelineStatus.LOADED
                    and current_params == new_params
                ):
                    is_loaded = True
            elif (
                self._pipeline_id == pipeline_id
                and self._status == PipelineStatus.LOADED
            ):
                # Check if load params match
                current_main_params = self._load_params or {}
                if current_main_params == new_params:
                    # Main pipeline is loaded, register it in _pipelines for chaining
                    if self._pipeline is not None:
                        self._pipelines[pipeline_id] = self._pipeline
                        self._pipeline_load_params[pipeline_id] = current_main_params
                        self._pipeline_statuses[pipeline_id] = PipelineStatus.LOADED
                        is_loaded = True

            if is_loaded:
                logger.info(
                    f"Pipeline {pipeline_id} already loaded with matching parameters"
                )
                return True

            # If already loading, wait
            if self._pipeline_statuses.get(pipeline_id) == PipelineStatus.LOADING:
                logger.info(f"Pipeline {pipeline_id} already loading by another thread")
                return False

            # Mark as loading
            self._pipeline_statuses[pipeline_id] = PipelineStatus.LOADING

        # Release lock during slow loading operation
        logger.info(f"Loading pipeline: {pipeline_id}")

        try:
            # Load the pipeline synchronously
            pipeline = self._load_pipeline_implementation(pipeline_id, load_params)

            # Hold lock while updating state
            with self._lock:
                self._pipelines[pipeline_id] = pipeline
                self._pipeline_load_params[pipeline_id] = load_params or {}
                self._pipeline_statuses[pipeline_id] = PipelineStatus.LOADED

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
                self._pipeline_statuses[pipeline_id] = PipelineStatus.ERROR
                if pipeline_id in self._pipelines:
                    del self._pipelines[pipeline_id]
                if pipeline_id in self._pipeline_load_params:
                    del self._pipeline_load_params[pipeline_id]

            return False

    def get_status_info(self) -> dict[str, Any]:
        """Get detailed status information (thread-safe).

        Note: If status is ERROR, the error message is returned once and then cleared
        to prevent persistence across page reloads.

        Returns "loading" if any pipeline is loading,
        "error" if any pipeline has an error, and "loaded" only if all pipelines are loaded.
        """
        with self._lock:
            # Check status of all tracked pipelines
            has_loading = False
            has_error = False
            all_loaded = True

            # Check all tracked pipeline statuses
            for _pipeline_id, status in self._pipeline_statuses.items():
                if status == PipelineStatus.LOADING:
                    has_loading = True
                elif status == PipelineStatus.ERROR:
                    has_error = True
                elif status != PipelineStatus.LOADED:
                    all_loaded = False

            # Determine overall status
            if has_loading:
                overall_status = PipelineStatus.LOADING
            elif has_error:
                overall_status = PipelineStatus.ERROR
            elif all_loaded and len(self._pipelines) > 0:
                overall_status = PipelineStatus.LOADED
            else:
                overall_status = PipelineStatus.NOT_LOADED

            # Capture current state before clearing
            current_status = overall_status
            # Use load_params from first pipeline or stored load_params
            load_params = self._load_params
            if not load_params and self._pipeline_load_params:
                # Get load_params from first tracked pipeline
                load_params = next(iter(self._pipeline_load_params.values()))

            # Get pipeline_id from first loaded pipeline (for backward compatibility with API response)
            pipeline_id = None
            if self._pipelines:
                pipeline_id = next(iter(self._pipelines.keys()))

            # Capture loaded LoRA adapters from all pipelines
            # Collect from all pipelines that expose this attribute
            loaded_lora_adapters = None
            if self._pipelines:
                all_adapters = []
                seen_paths = set()

                for pipeline in self._pipelines.values():
                    if hasattr(pipeline, "loaded_lora_adapters"):
                        adapters = getattr(pipeline, "loaded_lora_adapters", None)
                        if adapters:
                            # Add adapters, avoiding duplicates by path
                            for adapter in adapters:
                                adapter_path = adapter.get("path")
                                if adapter_path and adapter_path not in seen_paths:
                                    all_adapters.append(adapter)
                                    seen_paths.add(adapter_path)

                if all_adapters:
                    loaded_lora_adapters = all_adapters

            # If there's an error, clear error statuses after capturing them
            # This ensures errors don't persist across page reloads
            if overall_status == PipelineStatus.ERROR:
                # Clear error statuses from tracked pipelines
                for pid in list(self._pipeline_statuses.keys()):
                    if self._pipeline_statuses[pid] == PipelineStatus.ERROR:
                        self._pipeline_statuses[pid] = PipelineStatus.NOT_LOADED

            # Error messages are logged but not tracked per-pipeline
            # The ERROR status is sufficient for the frontend
            combined_error = None

            # Return the captured state (with error status if it was an error)
            return {
                "status": current_status.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
                "loaded_lora_adapters": loaded_lora_adapters,
                "error": combined_error,
            }

    async def get_pipeline_async(self):
        """Get the loaded pipeline instance (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_pipeline)

    async def get_status_info_async(self) -> dict[str, Any]:
        """Get detailed status information (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_status_info)

    async def load_pipelines(
        self, pipeline_ids: list[str], load_params: dict | None = None
    ) -> bool:
        """
        Load multiple pipelines asynchronously.

        Args:
            pipeline_ids: List of pipeline IDs to load
            load_params: Pipeline-specific load parameters (applies to all pipelines)

        Returns:
            bool: True if all pipelines loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._load_pipelines_sync, pipeline_ids, load_params
        )

    def _load_pipelines_sync(
        self, pipeline_ids: list[str], load_params: dict | None = None
    ) -> bool:
        """Synchronous wrapper for loading multiple pipelines."""
        if not pipeline_ids:
            logger.error("No pipeline IDs provided")
            return False

        logger.info(f"Loading {len(pipeline_ids)} pipeline(s): {pipeline_ids}")

        # Store load_params for use by frame processor
        with self._lock:
            self._load_params = load_params

            # Identify pipelines that need to be unloaded:
            # 1. Currently loaded but not in new list
            # 2. In new list but with different load_params (e.g., different resolution)
            currently_loaded = set(self._pipelines.keys())
            # Also check main pipeline if it exists
            if self._pipeline_id and self._pipeline_id not in currently_loaded:
                currently_loaded.add(self._pipeline_id)

            new_params = load_params or {}
            pipelines_to_unload = set()

            for loaded_id in currently_loaded:
                # Unload if pipeline not in new list or if load_params changed
                current_params = self._pipeline_load_params.get(loaded_id, {})
                if loaded_id not in pipeline_ids or current_params != new_params:
                    pipelines_to_unload.add(loaded_id)

            # Unload pipelines that need to be unloaded
            for pipeline_id_to_unload in pipelines_to_unload:
                self._unload_pipeline_by_id_unsafe(pipeline_id_to_unload)

        # Load all pipelines
        success = True
        for pipeline_id in pipeline_ids:
            try:
                result = self._load_pipeline_by_id_sync(pipeline_id, load_params)
                if not result:
                    logger.error(f"Failed to load pipeline: {pipeline_id}")
                    success = False
            except Exception as e:
                logger.error(f"Error loading pipeline {pipeline_id}: {e}")
                success = False

        if success:
            logger.info(f"All {len(pipeline_ids)} pipeline(s) loaded successfully")
        else:
            logger.error("Some pipelines failed to load")

        return success

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
        """Extract and apply common load parameters (resolution, seed, LoRAs, VAE type) to config.

        Args:
            config: Pipeline config dict to update
            load_params: Load parameters dict (may contain height, width, seed, loras, lora_merge_mode, vae_type)
            default_height: Default height if not in load_params
            default_width: Default width if not in load_params
            default_seed: Default seed if not in load_params
        """
        height = default_height
        width = default_width
        seed = default_seed
        loras = None
        lora_merge_mode = "permanent_merge"
        vae_type = "wan"  # Default VAE type

        if load_params:
            height = load_params.get("height", default_height)
            width = load_params.get("width", default_width)
            seed = load_params.get("seed", default_seed)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)
            vae_type = load_params.get("vae_type", vae_type)

        config["height"] = height
        config["width"] = width
        config["seed"] = seed
        config["vae_type"] = vae_type
        if loras:
            config["loras"] = loras
        # Pass merge_mode directly to mixin, not via config
        config["_lora_merge_mode"] = lora_merge_mode

    def unload_pipeline_by_id(self, pipeline_id: str):
        """Unload a specific pipeline by ID (thread-safe)."""
        with self._lock:
            self._unload_pipeline_by_id_unsafe(pipeline_id)

    def unload_all_pipelines(self):
        """Unload all pipelines (thread-safe)."""
        with self._lock:
            # Get all pipeline IDs to unload (from tracked pipelines and main pipeline)
            pipeline_ids_to_unload = set(self._pipelines.keys())
            if self._pipeline_id and self._pipeline_id not in pipeline_ids_to_unload:
                pipeline_ids_to_unload.add(self._pipeline_id)

            # Unload all pipelines
            for pipeline_id in pipeline_ids_to_unload:
                self._unload_pipeline_by_id_unsafe(pipeline_id)

    def _unload_pipeline_by_id_unsafe(self, pipeline_id: str):
        """Unload a specific pipeline by ID. Must be called with lock held."""
        # Check if pipeline exists (either in _pipelines or as main pipeline)
        pipeline_exists = (
            pipeline_id in self._pipelines or self._pipeline_id == pipeline_id
        )
        if not pipeline_exists:
            return

        logger.info(f"Unloading pipeline: {pipeline_id}")

        # Remove from tracked pipelines
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
        if pipeline_id in self._pipeline_statuses:
            del self._pipeline_statuses[pipeline_id]
        if pipeline_id in self._pipeline_load_params:
            del self._pipeline_load_params[pipeline_id]

        # If this was the main pipeline, also clear main pipeline state
        if self._pipeline_id == pipeline_id:
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
        from scope.core.pipelines.registry import PipelineRegistry

        # Check if pipeline is in registry
        pipeline_class = PipelineRegistry.get(pipeline_id)

        # List of built-in pipelines with custom initialization
        BUILTIN_PIPELINES = {
            "streamdiffusionv2",
            "passthrough",
            "longlive",
            "krea-realtime-video",
            "reward-forcing",
            "memflow",
            "video-depth-anything",
            "controller-viz",
            "rife",
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

        elif pipeline_id == "video-depth-anything":
            from scope.core.pipelines import VideoDepthAnythingPipeline

            # Create minimal config - pipeline handles its own model paths
            config = OmegaConf.create({})

            pipeline = VideoDepthAnythingPipeline(
                config,
                device=get_device(),
                dtype=torch.float16,
            )
            logger.info("VideoDepthAnything pipeline initialized")
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
                }
            )

            # Configure VACE support if enabled in load_params (default: True)
            vace_enabled = True
            if load_params:
                vace_enabled = load_params.get("vace_enabled", True)

            if vace_enabled:
                # Use 14B VACE checkpoint for Krea (not the default 1.3B from _configure_vace)
                config["vace_path"] = str(
                    get_model_file_path(
                        "WanVideo_comfy/Wan2_1-VACE_module_14B_bf16.safetensors"
                    )
                )
                logger.debug(
                    f"Krea: Using 14B VACE checkpoint at {config['vace_path']}"
                )
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

        elif pipeline_id == "video-depth-anything":
            from scope.core.pipelines import VideoDepthAnythingPipeline

            # Create config from load_params
            config = OmegaConf.create({})

            # Apply load parameters (resolution) to config
            # Note: video-depth-anything doesn't use height/width from load_params,
            # but we apply them anyway for consistency (they'll be ignored)
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
            )

            # Add video-depth-anything-specific parameters
            if load_params:
                config.input_size = load_params.get(
                    "input_size", 518
                )  # Default 518 (optimal for model)
                config.fp32 = load_params.get("fp32", False)
            else:
                config.input_size = 518
                config.fp32 = False

            pipeline = VideoDepthAnythingPipeline(
                config,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                dtype=torch.float16,
            )
            logger.info("VideoDepthAnything pipeline initialized")
            return pipeline

        elif pipeline_id == "controller-viz":
            from scope.core.pipelines import ControllerVisualizerPipeline

            # Use load parameters for resolution, default to 512x512
            height = 512
            width = 512
            if load_params:
                height = load_params.get("height", 512)
                width = load_params.get("width", 512)

            pipeline = ControllerVisualizerPipeline(
                height=height,
                width=width,
                device=get_device(),
                dtype=torch.float32,
            )
            logger.info("ControllerVisualizer pipeline initialized")
            return pipeline
        elif pipeline_id == "rife":
            from scope.core.pipelines import RIFEPipeline

            # Create minimal config - RIFE pipeline handles its own model paths via artifacts
            config = OmegaConf.create({})

            # Apply load parameters (resolution, seed) to config for consistency
            # Note: RIFE doesn't use these parameters but we apply them for consistency
            self._apply_load_params(
                config,
                load_params,
                default_height=512,
                default_width=512,
                default_seed=42,
            )

            pipeline = RIFEPipeline(
                config,
                device=get_device(),
                dtype=torch.float16,
            )
            logger.info("RIFE pipeline initialized")
            return pipeline
        else:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED
