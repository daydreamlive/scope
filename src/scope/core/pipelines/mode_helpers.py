"""Universal input modes helpers for pipelines.

This module provides a mixin class with helpers for pipelines that support
both text-to-video and video-to-video generation modes.
"""

from typing import Any

import torch

from .interface import Requirements


class UniversalInputModesMixin:
    """Mixin providing universal input mode support (text and video modes).

    This mixin adds helpers for pipelines that support both text-to-video
    and video-to-video generation, including:
    - Determining video input requirements based on mode
    - Lazy loading mode-specific components (e.g., VAEs)
    - Selecting appropriate block graphs per mode
    - Managing VAE initialization and caching
    """

    def prepare(
        self, generation_mode: str | None = None, **kwargs
    ) -> Requirements | None:
        """Determine whether this call should consume video input.

        This implementation uses mode configuration to decide if video input
        is required. Pipelines can override this for custom behavior.

        Args:
            generation_mode: Explicit mode override (text/video), or None for native mode
            **kwargs: Additional parameters that may contain generation_mode

        Returns:
            Requirements object specifying input_size, or None if no input required
        """
        from .defaults import (
            GENERATION_MODE_VIDEO,
            get_mode_config,
            resolve_generation_mode,
        )

        mode = resolve_generation_mode(generation_mode, kwargs, self.__class__)

        if mode == GENERATION_MODE_VIDEO:
            mode_config = get_mode_config(self.__class__, mode)
            # Extract from JSON Schema object
            input_size_schema = mode_config.get("input_size")
            if input_size_schema is not None:
                input_size = input_size_schema.get("default")
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
        # Extract from JSON Schema object
        strategy_schema = mode_config.get(strategy_key)
        strategy = strategy_schema.get("default") if strategy_schema else None

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

        Uses a dictionary-based mapping that can be extended via overriding
        _get_mode_to_blocks_mapping() for custom modes without modifying this method.

        Args:
            mode: Current generation mode (text/video)

        Returns:
            Block graph instance for the specified mode
        """
        import logging

        from .defaults import GENERATION_MODE_TEXT

        logger = logging.getLogger(__name__)

        mapping = self._get_mode_to_blocks_mapping()

        if mode not in mapping:
            logger.warning(
                f"_select_blocks_for_mode: Unknown mode '{mode}', falling back to {GENERATION_MODE_TEXT}"
            )
            mode = GENERATION_MODE_TEXT

        return mapping[mode]

    def _get_mode_to_blocks_mapping(self) -> dict[str, Any]:
        """Return mapping of mode names to block graphs.

        Override this in pipeline subclasses to support custom modes beyond
        the standard text/video modes.

        Returns:
            Dictionary mapping mode names to block graph instances

        Example:
            def _get_mode_to_blocks_mapping(self):
                from .defaults import GENERATION_MODE_VIDEO, GENERATION_MODE_TEXT
                return {
                    GENERATION_MODE_VIDEO: self.blocks_video,
                    GENERATION_MODE_TEXT: self.blocks_text,
                    "hybrid": self.blocks_hybrid,  # Custom mode
                }
        """
        from .defaults import GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO

        return {
            GENERATION_MODE_VIDEO: self.blocks_video,
            GENERATION_MODE_TEXT: self.blocks_text,
        }

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

    def _initialize_pipeline_state(
        self,
        config: Any,
        generator: Any,
        text_encoder: Any,
        blocks_text: Any,
        blocks_video: Any,
        model_config: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize pipeline state and components from configuration.

        This helper consolidates the common pattern of initializing pipeline
        components, state, and applying mode-specific defaults. This reduces
        boilerplate across pipeline implementations.

        Args:
            config: Configuration object with pipeline parameters
            generator: Diffusion model wrapper
            text_encoder: Text encoder wrapper
            blocks_text: Modular blocks for text-to-video mode
            blocks_video: Modular blocks for video-to-video mode
            model_config: Model configuration with additional parameters
            device: Target device
            dtype: Target dtype

        Side effects:
            Sets self.components, self.state, self.blocks_text, self.blocks_video,
            and self.first_call attributes.
        """
        from diffusers.modular_pipelines import PipelineState

        from .blending import EmbeddingBlender
        from .components import ComponentsManager
        from .defaults import get_mode_config
        from .helpers import initialize_state_from_config

        # Create components config
        components_config = {}
        components_config.update(model_config)
        components_config["device"] = device
        components_config["dtype"] = dtype

        components = ComponentsManager(components_config)
        components.add("generator", generator)
        components.add("scheduler", generator.get_scheduler())
        components.add("text_encoder", text_encoder)

        embedding_blender = EmbeddingBlender(device=device, dtype=dtype)
        components.add("embedding_blender", embedding_blender)

        # Store block graphs and components
        self.blocks_text = blocks_text
        self.blocks_video = blocks_video
        self.components = components
        self.state = PipelineState()

        # Initialize state with native mode defaults
        native_mode_config = get_mode_config(self.__class__)
        initialize_state_from_config(self.state, config, native_mode_config)

        self.first_call = True

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
        from .defaults import apply_mode_defaults_to_state, resolve_generation_mode
        from .process import postprocess_chunk

        mode = resolve_generation_mode(
            state.get("generation_mode"), kwargs, self.__class__
        )

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
