"""Base interface for all pipelines."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int



class Pipeline(ABC):
    """Abstract base class for all pipelines.

    Pipelines must implement get_schema() to return their metadata and configuration.
    This metadata is used for:
    - Dynamic UI generation
    - API introspection
    - Parameter validation
    - Mode-specific configuration

    See helpers.py and schema.py for utilities to simplify schema creation.
    """

    @classmethod
    @abstractmethod
    def get_schema(cls) -> dict[str, Any]:
        """Return complete pipeline schema with metadata and mode configurations.

        The schema should be a dictionary compatible with JSON Schema and OpenAPI
        conventions, containing:
        - id: Unique pipeline identifier
        - name: Human-readable pipeline name
        - description: Pipeline capabilities description
        - version: Pipeline version string
        - native_mode: Native generation mode ("text" or "video")
        - supported_modes: List of supported generation modes
        - mode_configs: Dict mapping mode names to their parameter configurations

        Returns:
            Complete pipeline schema dictionary

        Example:
            from .helpers import build_pipeline_schema
            from .defaults import GENERATION_MODE_VIDEO

            return build_pipeline_schema(
                pipeline_id="my-pipeline",
                name="My Pipeline",
                description="Description of capabilities",
                native_mode=GENERATION_MODE_VIDEO,
                shared={"manage_cache": True, "base_seed": 42},
                text_overrides={
                    "resolution": {"height": 512, "width": 512},
                    "denoising_steps": [1000, 750],
                },
                video_overrides={
                    "resolution": {"height": 512, "width": 512},
                    "denoising_steps": [750, 250],
                    "noise_scale": 0.7,
                },
            )
        """
        pass

    def prepare(
        self, generation_mode: str | None = None, **kwargs
    ) -> Requirements | None:
        """Determine whether this call should consume video input.

        This base implementation uses mode configuration to decide if video input
        is required. Pipelines can override this for custom behavior.

        Args:
            generation_mode: Explicit mode override (text/video), or None for native mode
            **kwargs: Additional parameters that may contain generation_mode

        Returns:
            Requirements object specifying input_size, or None if no input required
        """
        from .defaults import GENERATION_MODE_VIDEO, get_mode_config

        schema = self.__class__.get_schema()
        native_mode = schema["native_mode"]
        mode = generation_mode or kwargs.get("generation_mode") or native_mode

        if mode == GENERATION_MODE_VIDEO:
            mode_config = get_mode_config(self.__class__, mode)
            input_size = mode_config.get("input_size")
            if input_size is not None:
                return Requirements(input_size=input_size)

        return None

    def _get_component_for_mode(
        self,
        component_name: str,
        mode: str,
        factory_func: callable,
        cache_attr: str | None = None,
        **factory_kwargs,
    ) -> Any:
        """Get or create a mode-specific component with lazy loading.

        This enables pipelines to use different component implementations per mode
        without loading all variants upfront. Commonly used for VAEs that differ
        between text and video generation modes.

        Args:
            component_name: Name of the component (e.g. "vae")
            mode: Current generation mode (text/video)
            factory_func: Function to create the component if not cached
            cache_attr: Optional attribute name for caching (e.g. "_vae_cache")
            **factory_kwargs: Additional kwargs passed to factory_func

        Returns:
            Component instance for the specified mode

        Example:
            vae = self._get_component_for_mode(
                component_name="vae",
                mode=mode,
                factory_func=create_vae,
                cache_attr="_vae_cache",
                model_dir=model_dir,
                device=device,
                dtype=dtype
            )
            self.components.add("vae", vae)
        """
        from .defaults import get_mode_config

        mode_config = get_mode_config(self.__class__, mode)

        strategy_key = f"{component_name}_strategy"
        strategy = mode_config.get(strategy_key)

        if cache_attr:
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            cache = getattr(self, cache_attr)

            cache_key = (mode, strategy)
            if cache_key in cache:
                return cache[cache_key]

        schema = self.__class__.get_schema()
        pipeline_name = schema["id"]

        component = factory_func(
            strategy=strategy,
            pipeline_name=pipeline_name,
            **factory_kwargs,
        )

        if cache_attr:
            cache[cache_key] = component

        return component

    def _select_blocks_for_mode(self, mode: str) -> Any:
        """Select appropriate block graph based on generation mode.

        Args:
            mode: Current generation mode (text/video)

        Returns:
            Block graph instance for the specified mode
        """
        from .defaults import GENERATION_MODE_VIDEO

        if mode == GENERATION_MODE_VIDEO:
            return self.blocks_video
        return self.blocks_text

    def _init_vae_lazy_loading(
        self, device: torch.device, dtype: torch.dtype, **vae_init_kwargs
    ) -> None:
        """Initialize VAE lazy loading infrastructure.

        This helper consolidates the pattern of storing VAE initialization parameters
        for later lazy loading based on generation mode.

        Args:
            device: Target device for VAE
            dtype: Target dtype for VAE
            **vae_init_kwargs: VAE initialization kwargs to store for lazy loading
        """
        self._vae_init_kwargs = vae_init_kwargs
        self._vae_device = device
        self._vae_dtype = dtype
        self._vae_cache = {}

    def _prepare_and_execute_blocks(
        self, state: Any, components: Any, **kwargs
    ) -> torch.Tensor:
        """Prepare VAE component and execute pipeline blocks based on generation mode.

        This helper consolidates the common pattern of:
        1. Applying mode-specific defaults to state
        2. Loading the appropriate VAE for the mode
        3. Selecting and executing the correct block graph
        4. Post-processing the output

        Args:
            state: Pipeline state object
            components: Components manager
            **kwargs: Generation parameters (may include generation_mode)

        Returns:
            Post-processed output tensor
        """
        from .base.vae import create_vae
        from .defaults import apply_mode_defaults_to_state
        from .process import postprocess_chunk

        schema = self.__class__.get_schema()
        native_mode = schema["native_mode"]
        mode = state.get("generation_mode") or native_mode

        apply_mode_defaults_to_state(state, self.__class__, mode, kwargs)

        vae = self._get_component_for_mode(
            component_name="vae",
            mode=mode,
            factory_func=create_vae,
            cache_attr="_vae_cache",
            **self._vae_init_kwargs,
        )
        vae = vae.to(device=self._vae_device, dtype=self._vae_dtype)
        components.add("vae", vae)

        blocks = self._select_blocks_for_mode(mode)

        _, state = blocks(components, state)
        return postprocess_chunk(state.values["output_video"])

    @abstractmethod
    def __call__(
        self, input: torch.Tensor | list[torch.Tensor] | None = None, **kwargs
    ) -> torch.Tensor:
        """
        Process a chunk of video frames.

        Args:
            input: A tensor in BCTHW format OR a list of frame tensors in THWC format (in [0, 255] range), or None
            **kwargs: Additional parameters

        Returns:
            A processed chunk tensor in THWC format and [0, 1] range
        """
        pass
