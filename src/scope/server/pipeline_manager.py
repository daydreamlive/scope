"""Pipeline Manager for lazy loading and managing ML pipelines."""

import asyncio
import gc
import logging
import threading
import time
from enum import Enum
from typing import Any

import torch
from omegaconf import OmegaConf

from .kafka_publisher import publish_event

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

        # Support for multiple pipelines (for pipeline chaining / graph)
        # Keyed by instance_key (e.g. "longlive", "longlive:1" for duplicates)
        self._pipelines: dict[str, Any] = {}  # instance_key -> pipeline instance
        self._pipeline_statuses: dict[
            str, PipelineStatus
        ] = {}  # instance_key -> status
        self._pipeline_load_params: dict[str, dict] = {}  # instance_key -> load_params
        self._pipeline_registry_ids: dict[
            str, str
        ] = {}  # instance_key -> pipeline_id (registry key)
        self._load_events: dict[
            str, threading.Event
        ] = {}  # instance_key -> load completion event

        # Loading stage for frontend display (e.g., "Loading diffusion model...")
        self._loading_stage: str | None = None

        # Set to True if torch._dynamo.reset() failed during an unload; stale
        # Dynamo/FP8 compile caches may still be present, so force compile=False
        # on all subsequent pipeline loads until the worker process restarts.
        self._dynamo_reset_failed: bool = False

    def set_loading_stage(self, stage: str | None) -> None:
        """Set the current loading stage (thread-safe)."""
        with self._lock:
            self._loading_stage = stage

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

    def get_loaded_pipeline_ids(self) -> list[str]:
        """Return the IDs of all currently loaded pipelines (thread-safe)."""
        with self._lock:
            return [
                pid
                for pid, status in self._pipeline_statuses.items()
                if status == PipelineStatus.LOADED
            ]

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
        self,
        pipeline_id: str,
        load_params: dict | None = None,
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        """
        Load a pipeline by ID asynchronously (private method).

        Args:
            pipeline_id: ID of pipeline to load
            load_params: Pipeline-specific load parameters
            connection_info: Optional connection info (gpu_type, fal_host) for event correlation
            user_id: Optional user ID for event correlation

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_pipeline_by_id_sync,
            pipeline_id,
            load_params,
            connection_id,
            connection_info,
            user_id,
        )

    def _load_pipeline_by_id_sync(
        self,
        pipeline_id: str,
        load_params: dict | None = None,
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
        instance_key: str | None = None,
    ) -> bool:
        """Synchronous wrapper for loading a pipeline by ID.

        Args:
            pipeline_id: Registry key used to look up the pipeline class.
            load_params: Parameters for pipeline initialization.
            instance_key: Unique storage key. Defaults to ``pipeline_id``
                when there is only one instance of a given pipeline.
        """
        key = instance_key or pipeline_id
        with self._lock:
            # Check if already loaded with same params
            current_params = self._pipeline_load_params.get(key, {})
            new_params = load_params or {}

            # Check if pipeline is already loaded (either in _pipelines or as main pipeline)
            is_loaded = False
            if key in self._pipelines:
                if (
                    self._pipeline_statuses.get(key) == PipelineStatus.LOADED
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
                        self._pipelines[key] = self._pipeline
                        self._pipeline_load_params[key] = current_main_params
                        self._pipeline_registry_ids[key] = pipeline_id
                        self._pipeline_statuses[key] = PipelineStatus.LOADED
                        is_loaded = True

            if is_loaded:
                logger.info(f"Pipeline {key} already loaded with matching parameters")
                return True

            # If already loading, wait for it to complete
            if self._pipeline_statuses.get(key) == PipelineStatus.LOADING:
                logger.info(
                    f"Pipeline {key} already loading by another thread, waiting..."
                )
                load_event = self._load_events.get(key)
                if load_event:
                    # Release lock while waiting
                    self._lock.release()
                    try:
                        # Wait up to 5 minutes for load to complete
                        load_event.wait(timeout=300)
                    finally:
                        self._lock.acquire()

                    # Check if pipeline is now loaded
                    if self._pipeline_statuses.get(key) == PipelineStatus.LOADED:
                        logger.info(f"Pipeline {key} loaded by another thread")
                        return True
                    else:
                        logger.warning(
                            f"Pipeline {key} load by another thread did not succeed, "
                            f"status: {self._pipeline_statuses.get(key)}"
                        )
                        return False
                else:
                    # No event found (shouldn't happen), fall through to load
                    logger.warning(
                        f"Pipeline {key} marked as LOADING but no event found"
                    )

            # Mark as loading and create event for waiters
            self._pipeline_statuses[key] = PipelineStatus.LOADING
            load_event = threading.Event()
            self._load_events[key] = load_event

        # Release lock during slow loading operation
        logger.info(f"Loading pipeline: {key} (registry: {pipeline_id})")
        logger.info("Initial load params: %s", load_params or {})

        # Publish pipeline_load_start event
        load_start_time = time.monotonic()
        publish_event(
            event_type="pipeline_load_start",
            connection_id=connection_id,
            connection_info=connection_info,
            pipeline_ids=[pipeline_id],
            user_id=user_id,
            metadata={"load_params": load_params} if load_params else None,
        )

        try:
            # Load the pipeline synchronously
            self.set_loading_stage("Initializing pipeline...")
            pipeline = self._load_pipeline_implementation(
                pipeline_id, load_params, stage_callback=self.set_loading_stage
            )

            # Hold lock while updating state
            self.set_loading_stage(None)
            with self._lock:
                self._pipelines[key] = pipeline
                self._pipeline_load_params[key] = load_params or {}
                self._pipeline_registry_ids[key] = pipeline_id
                self._pipeline_statuses[key] = PipelineStatus.LOADED
                # Signal waiters that load is complete
                if key in self._load_events:
                    self._load_events[key].set()

            logger.info(f"Pipeline {key} loaded successfully")

            # Publish pipeline_loaded event with load duration
            load_duration_ms = int((time.monotonic() - load_start_time) * 1000)
            metadata = {
                "load_duration_ms": load_duration_ms,
                "load_start_time_ms": load_start_time,
            }
            if load_params:
                metadata["load_params"] = load_params
            publish_event(
                event_type="pipeline_loaded",
                connection_id=connection_id,
                connection_info=connection_info,
                pipeline_ids=[pipeline_id],
                user_id=user_id,
                metadata=metadata,
            )
            return True

        except Exception as e:
            self.set_loading_stage(None)
            from .models_config import get_models_dir

            models_dir = get_models_dir()
            error_msg = f"Failed to load pipeline {key}: {e}"
            logger.error(
                f"{error_msg}. If this error persists, consider removing the models "
                f"directory '{models_dir}' and re-downloading models."
            )

            # Hold lock while updating state with error
            with self._lock:
                self._pipeline_statuses[key] = PipelineStatus.ERROR
                if key in self._pipelines:
                    del self._pipelines[key]
                if key in self._pipeline_load_params:
                    del self._pipeline_load_params[key]
                if key in self._pipeline_registry_ids:
                    del self._pipeline_registry_ids[key]
                # Signal waiters that load is complete (even though it failed)
                if key in self._load_events:
                    self._load_events[key].set()

            # Publish error event for pipeline load failure
            publish_event(
                event_type="error",
                connection_id=connection_id,
                connection_info=connection_info,
                pipeline_ids=[pipeline_id],
                user_id=user_id,
                error={
                    "error_type": "pipeline_load_failed",
                    "message": str(e),
                    "exception_type": type(e).__name__,
                    "recoverable": True,
                },
            )

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

            # Determine media modalities from the loaded pipeline chain.
            # Use registry IDs (not instance keys) so the lookup works in
            # graph mode where instance keys are node IDs like "pipeline_1".
            from scope.core.pipelines.registry import PipelineRegistry

            registry_ids = [
                self._pipeline_registry_ids.get(key, key) for key in self._pipelines
            ]
            produces_video = PipelineRegistry.chain_produces_video(registry_ids)
            produces_audio = PipelineRegistry.chain_produces_audio(registry_ids)

            # Return the captured state (with error status if it was an error)
            return {
                "status": current_status.value,
                "pipeline_id": pipeline_id,
                "load_params": load_params,
                "loaded_lora_adapters": loaded_lora_adapters,
                "error": combined_error,
                "loading_stage": self._loading_stage,
                "produces_video": produces_video,
                "produces_audio": produces_audio,
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
        self,
        pipelines: list[tuple[str, str, dict | None]],
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Load multiple pipelines asynchronously.

        Args:
            pipelines: List of (node_id, pipeline_id, load_params) tuples.
                node_id is the graph node ID used as the instance key.
                The same pipeline_id may appear more than once; each entry
                produces a separate instance keyed by its node_id.
            connection_id: Optional connection ID for event correlation.
            connection_info: Optional connection info for event correlation.
            user_id: Optional user ID for event correlation.

        Returns:
            bool: True if all pipelines loaded successfully, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._load_pipelines_sync,
            pipelines,
            connection_id,
            connection_info,
            user_id,
        )

    def _is_pipeline_loaded_with(
        self, key: str, pipeline_id: str, params: dict
    ) -> bool:
        """Check if the pipeline at *key* is loaded and matches the given type and params.

        Must be called with ``self._lock`` held.
        """
        return (
            key in self._pipelines
            and self._pipeline_statuses.get(key) == PipelineStatus.LOADED
            and self._pipeline_registry_ids.get(key) == pipeline_id
            and self._pipeline_load_params.get(key, {}) == params
        )

    def _rekey_pipeline(self, old_key: str, new_key: str) -> None:
        """Move a pipeline instance from *old_key* to *new_key* in all tracking dicts.

        Must be called with ``self._lock`` held.
        """
        self._pipelines[new_key] = self._pipelines.pop(old_key)
        self._pipeline_load_params[new_key] = self._pipeline_load_params.pop(old_key)
        self._pipeline_registry_ids[new_key] = self._pipeline_registry_ids.pop(old_key)
        self._pipeline_statuses[new_key] = self._pipeline_statuses.pop(old_key)

    def _find_reusable_pipeline(
        self,
        node_id: str,
        pipeline_id: str,
        params: dict,
        claimed_keys: set[str],
        reserved_keys: set[str],
    ) -> str | None:
        """Find an existing loaded pipeline that can be re-keyed to *node_id*.

        Skips keys that are already claimed or reserved by other new entries
        (to avoid stealing a key that another entry needs at that exact position).

        Must be called with ``self._lock`` held.
        """
        for old_key in self._pipelines:
            if old_key in claimed_keys:
                continue
            if old_key in reserved_keys and old_key != node_id:
                continue
            if self._is_pipeline_loaded_with(old_key, pipeline_id, params):
                return old_key
        return None

    def _load_pipelines_sync(
        self,
        pipelines: list[tuple[str, str, dict | None]],
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Synchronous wrapper for loading multiple pipelines."""
        if not pipelines:
            logger.error("No pipelines provided")
            return False

        node_ids = [node_id for node_id, _, _ in pipelines]
        pipeline_ids = [pid for _, pid, _ in pipelines]
        logger.info(
            f"Loading {len(pipelines)} pipeline(s): {list(zip(node_ids, pipeline_ids, strict=True))}"
        )

        # Store first load_params for backward-compat status reporting
        with self._lock:
            self._load_params = pipelines[0][2] if pipelines else None

            new_key_set = set(node_ids)

            # --- Phase 1: Match requested pipelines to already-loaded instances ---
            #
            # Goal: avoid expensive unload + reload when we can reuse an
            # existing pipeline instance.  Each requested entry is matched:
            #
            #   1. Exact match — same node_id already holds a compatible instance.
            #   2. Re-key match — a compatible instance exists under a *different*
            #      key; rename it to the new node_id (cheap vs. full reload).
            #   3. No match — schedule a fresh load in Phase 3.
            #
            # NOTE: Even if two entries share the same pipeline_id and params,
            # each gets its own instance because we don't yet support multiple
            # graph nodes sharing a single pipeline instance.

            claimed_old_keys: set[str] = set()
            entries_needing_load: list[tuple[str, str, dict | None]] = []

            for node_id, pipeline_id, load_params in pipelines:
                params = load_params or {}

                # 1. Exact match — same key, same pipeline type and params.
                if self._is_pipeline_loaded_with(node_id, pipeline_id, params):
                    claimed_old_keys.add(node_id)
                    continue

                # 2. Re-key match — find a compatible instance under a different key.
                matched_key = self._find_reusable_pipeline(
                    node_id, pipeline_id, params, claimed_old_keys, new_key_set
                )
                if matched_key is not None:
                    logger.info(
                        f"Re-keying pipeline {pipeline_id} "
                        f"from '{matched_key}' to '{node_id}'"
                    )
                    self._rekey_pipeline(matched_key, node_id)
                    claimed_old_keys.add(matched_key)
                    continue

                # 3. No match — needs a fresh load.
                entries_needing_load.append((node_id, pipeline_id, load_params))

            # --- Phase 2: Unload stale pipelines ---
            #
            # Any old instance that wasn't claimed in Phase 1 and isn't in the
            # new request set is no longer needed — free its resources.

            for old_key in list(self._pipelines.keys()):
                if old_key not in new_key_set and old_key not in claimed_old_keys:
                    self._unload_pipeline_by_id_unsafe(
                        old_key,
                        connection_id=connection_id,
                        connection_info=connection_info,
                        user_id=user_id,
                    )

            # Clean up stale status entries (e.g. from previously errored loads).
            for key in list(self._pipeline_statuses.keys()):
                if key not in new_key_set:
                    del self._pipeline_statuses[key]

        # Phase 3: Load new entries
        success = True
        for node_id, pipeline_id, load_params in entries_needing_load:
            try:
                result = self._load_pipeline_by_id_sync(
                    pipeline_id,
                    load_params,
                    connection_id=connection_id,
                    connection_info=connection_info,
                    user_id=user_id,
                    instance_key=node_id,
                )
                if not result:
                    logger.error(f"Failed to load pipeline: {node_id}")
                    success = False
            except Exception as e:
                logger.error(f"Error loading pipeline {node_id}: {e}")
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
            # Always apply vace_context_scale when provided (applies to both
            # ref-image and input-video VACE modes)
            config["vace_context_scale"] = load_params.get("vace_context_scale", 1.0)

            ref_images = load_params.get("ref_images", [])
            if ref_images:
                config["ref_images"] = ref_images
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
        """Extract and apply common load parameters

        Args:
            config: Pipeline config dict to update
            load_params: Load parameters dict (may contain height, width, base_seed, loras, lora_merge_mode, vae_type)
            default_height: Default height if not in load_params
            default_width: Default width if not in load_params
            default_seed: Default base_seed if not in load_params
        """
        height = default_height
        width = default_width
        base_seed = default_seed
        loras = None
        lora_merge_mode = "permanent_merge"
        vae_type = "wan"  # Default VAE type

        if load_params:
            height = load_params.get("height", default_height)
            width = load_params.get("width", default_width)
            base_seed = load_params.get("base_seed", default_seed)
            loras = load_params.get("loras", None)
            lora_merge_mode = load_params.get("lora_merge_mode", lora_merge_mode)
            vae_type = load_params.get("vae_type", vae_type)

        config["height"] = height
        config["width"] = width
        config["base_seed"] = base_seed
        config["vae_type"] = vae_type
        if loras:
            config["loras"] = loras
        # Pass merge_mode directly to mixin, not via config
        config["_lora_merge_mode"] = lora_merge_mode

    def unload_pipeline_by_id(
        self,
        pipeline_id: str,
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
    ):
        """Unload a specific pipeline by ID (thread-safe)."""
        with self._lock:
            self._unload_pipeline_by_id_unsafe(
                pipeline_id,
                connection_id=connection_id,
                connection_info=connection_info,
                user_id=user_id,
            )

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

    def _unload_pipeline_by_id_unsafe(
        self,
        pipeline_id: str,
        connection_id: str | None = None,
        connection_info: dict[str, Any] | None = None,
        user_id: str | None = None,
    ):
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
        if pipeline_id in self._pipeline_registry_ids:
            del self._pipeline_registry_ids[pipeline_id]
        if pipeline_id in self._load_events:
            del self._load_events[pipeline_id]

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

        # Reset torch.compile compilation cache to prevent stale compiled graphs
        # (especially those specialized for Float8Tensor weights) from leaking into
        # subsequently loaded pipelines. Without this, longlive's FP8-compiled graph
        # cache can corrupt Krea's compile attempt, causing as_strided dispatch errors.
        try:
            torch._dynamo.reset()
            logger.info("torch._dynamo cache reset")
        except Exception as e:
            logger.warning(
                f"torch._dynamo reset failed: {e}. "
                "Stale compile caches may remain in this worker; "
                "forcing compile=False for all subsequent pipeline loads."
            )
            self._dynamo_reset_failed = True

        # Publish pipeline_unloaded event
        publish_event(
            event_type="pipeline_unloaded",
            connection_id=connection_id,
            connection_info=connection_info,
            pipeline_ids=[pipeline_id],
            user_id=user_id,
        )

    def _load_pipeline_implementation(
        self,
        pipeline_id: str,
        load_params: dict | None = None,
        stage_callback=None,
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
            "scribble",
            "gray",
            "optical-flow",
        }

        if pipeline_class is not None and pipeline_id not in BUILTIN_PIPELINES:
            # Plugin pipeline - use schema defaults merged with load_params
            logger.info(f"Loading plugin pipeline: {pipeline_id}")
            if stage_callback:
                stage_callback("Initializing pipeline...")
            config_class = pipeline_class.get_config_class()
            # Get defaults from schema fields
            schema_defaults = {}
            for name, field in config_class.model_fields.items():
                if field.default is not None:
                    schema_defaults[name] = field.default
            # Merge: load_params override schema defaults
            merged_params = {**schema_defaults, **(load_params or {})}
            return pipeline_class(**merged_params)

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
                stage_callback=stage_callback,
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
                stage_callback=stage_callback,
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
                # Extract VACE parameters from load_params (mirrors _configure_vace)
                if load_params:
                    config["vace_context_scale"] = load_params.get(
                        "vace_context_scale", 1.0
                    )
                    ref_images = load_params.get("ref_images", [])
                    if ref_images:
                        config["ref_images"] = ref_images
            else:
                logger.info("VACE disabled by load_params, skipping VACE configuration")

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

            # Only compile diffusion model for hopper; skip if a prior
            # torch._dynamo.reset() failed (stale caches would cause a crash).
            _hopper_gpu = torch.cuda.is_available() and any(
                x in torch.cuda.get_device_name(0).lower()
                for x in ("h100", "hopper")
            )
            _should_compile = _hopper_gpu and not self._dynamo_reset_failed
            if _hopper_gpu and self._dynamo_reset_failed:
                logger.warning(
                    "torch._dynamo reset previously failed; disabling torch.compile "
                    "for krea-realtime-video to avoid stale-cache crash. "
                    "Restart the worker process to re-enable compilation."
                )
            pipeline = KreaRealtimeVideoPipeline(
                config,
                quantization=quantization,
                compile=_should_compile,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
                stage_callback=stage_callback,
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
                stage_callback=stage_callback,
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
                stage_callback=stage_callback,
            )
            logger.info("MemFlow pipeline initialized")
            return pipeline

        elif pipeline_id == "video-depth-anything":
            from scope.core.pipelines import VideoDepthAnythingPipeline

            if stage_callback:
                stage_callback("Initializing pipeline...")

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

            if stage_callback:
                stage_callback("Initializing pipeline...")

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

            if stage_callback:
                stage_callback("Initializing pipeline...")

            # Create minimal config - RIFE pipeline handles its own model paths via artifacts
            config = OmegaConf.create({})

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
        elif pipeline_id == "scribble":
            from scope.core.pipelines import ScribblePipeline

            if stage_callback:
                stage_callback("Initializing pipeline...")

            pipeline = ScribblePipeline(
                device=get_device(),
                dtype=torch.float16,
            )
            logger.info("Scribble pipeline initialized")
            return pipeline
        elif pipeline_id == "gray":
            from scope.core.pipelines import GrayPipeline

            if stage_callback:
                stage_callback("Initializing pipeline...")

            pipeline = GrayPipeline(
                device=get_device(),
            )
            logger.info("Gray pipeline initialized")
            return pipeline

        elif pipeline_id == "optical-flow":
            from scope.core.pipelines import OpticalFlowPipeline
            from scope.core.pipelines.optical_flow.schema import OpticalFlowConfig

            if stage_callback:
                stage_callback("Initializing pipeline...")

            # Create config with schema defaults, overridden by load_params
            params = load_params or {}
            config = OmegaConf.create(
                {
                    "model_size": params.get(
                        "model_size",
                        OpticalFlowConfig.model_fields["model_size"].default,
                    ),
                }
            )

            pipeline = OpticalFlowPipeline(
                config,
                device=get_device(),
            )
            logger.info("OpticalFlow pipeline initialized")
            return pipeline
        else:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded and ready (thread-safe)."""
        with self._lock:
            return self._status == PipelineStatus.LOADED
