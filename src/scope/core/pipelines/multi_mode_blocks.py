"""Self-configuring blocks for multi-mode pipelines.

This module provides blocks that work with input-based routing (AutoPipelineBlocks).
The resolved mode is derived from input presence and stored in state for blocks
that need mode-specific behavior (like VAE loading).
"""

import logging
from typing import Any

from diffusers.modular_pipelines import ModularPipelineBlocks

logger = logging.getLogger(__name__)


class ConfigureForModeBlock(ModularPipelineBlocks):
    """Block that logs the resolved mode for debugging.

    The mode is resolved from input presence (e.g., 'video' input triggers video mode)
    and stored in state as _resolved_mode by MultiModePipeline._execute().

    This block primarily serves as a logging checkpoint to track which mode
    is being used for a given generation call.

    Example:
        class MyWorkflow(SequentialPipelineBlocks):
            block_classes = [
                ConfigureForModeBlock,  # Logs resolved mode
                LoadComponentsBlock,    # Loads mode-specific components
                # ... rest of workflow
            ]
    """

    def __call__(self, components: Any, state: Any) -> tuple[Any, Any]:
        """Log resolved mode.

        Args:
            components: Components manager
            state: Pipeline state with _resolved_mode set

        Returns:
            Tuple of (components, state) unchanged
        """
        mode = state.get("_resolved_mode")

        if mode is None:
            logger.warning(
                "ConfigureForModeBlock: No _resolved_mode in state. "
                "This should be set by MultiModePipeline._execute()"
            )
            mode = "text"
            state.set("_resolved_mode", mode)

        return components, state


class LoadComponentsBlock(ModularPipelineBlocks):
    """Block that loads mode-appropriate components with lazy loading.

    This block:
    1. Gets resolved mode from state (_resolved_mode)
    2. Loads mode-specific components (currently VAE) based on component declarations
    3. Caches loaded components to avoid redundant loading
    4. Adds components to the components manager

    The mode is resolved from input presence by MultiModePipeline.
    Component specifications come from the pipeline's get_components() method.
    For mode-specific components, the declaration looks like:
        "vae": {
            "text": {"strategy": "text_vae_strategy"},
            "video": {"strategy": "video_vae_strategy"},
        }

    Example:
        class MyWorkflow(SequentialPipelineBlocks):
            block_classes = [
                ConfigureForModeBlock,
                LoadComponentsBlock,  # Loads VAE for resolved mode
                # ... blocks that use VAE
            ]
    """

    def __call__(self, components: Any, state: Any) -> tuple[Any, Any]:
        """Load components for resolved mode.

        Args:
            components: Components manager
            state: Pipeline state with _resolved_mode set

        Returns:
            Tuple of (components, state) with components loaded
        """
        mode = state.get("_resolved_mode")

        if mode is None:
            logger.warning(
                "LoadComponentsBlock: No _resolved_mode in state. "
                "Defaulting to 'text' mode."
            )
            mode = "text"

        # Get component specifications from components manager
        # The pipeline class is stored in components config
        pipeline_class = components.config.get("pipeline_class")

        if pipeline_class is None:
            logger.warning(
                "LoadComponentsBlock: No pipeline_class in components config. "
                "Cannot load mode-specific components."
            )
            return components, state

        component_specs = pipeline_class.get_components()

        # Load VAE if it has mode-specific configuration
        vae_spec = component_specs.get("vae")
        if isinstance(vae_spec, dict) and mode in vae_spec:
            self._load_vae_for_mode(components, state, mode, vae_spec)
        else:
            logger.debug(
                f"LoadComponentsBlock: VAE not mode-specific or mode '{mode}' "
                f"not in VAE spec"
            )

        return components, state

    def _load_vae_for_mode(
        self, components: Any, state: Any, mode: str, vae_spec: dict
    ) -> None:
        """Load VAE for specific mode with caching.

        Args:
            components: Components manager
            state: Pipeline state
            mode: Current generation mode
            vae_spec: VAE specification dict with mode-specific strategies
        """
        from .base.vae import create_vae

        # Check for VAE strategy override from config (user-selected VAE)
        vae_strategy_override = components.config.get("vae_strategy_override")

        if vae_strategy_override:
            # Use override strategy for all modes
            strategy = vae_strategy_override
        else:
            # Use default strategy from pipeline's get_components()
            mode_spec = vae_spec.get(mode, {})
            strategy = mode_spec.get("strategy")

        if not strategy:
            logger.warning(
                f"LoadComponentsBlock: No strategy specified for VAE in mode '{mode}'"
            )
            return

        # Check cache
        if not hasattr(components, "_vae_cache"):
            components._vae_cache = {}

        cache_key = (mode, strategy)
        if cache_key in components._vae_cache:
            vae = components._vae_cache[cache_key]
            logger.debug(
                f"LoadComponentsBlock: Using cached VAE for mode '{mode}' "
                f"with strategy '{strategy}'"
            )
        else:
            # Load VAE
            pipeline_name = components.config.get("pipeline_name", "unknown")
            device = components.config.get("device")
            dtype = components.config.get("dtype")

            # Get additional VAE init kwargs if stored
            vae_init_kwargs = components.config.get("vae_init_kwargs", {})

            logger.info(
                f"LoadComponentsBlock: Loading VAE for mode '{mode}' "
                f"with strategy '{strategy}'"
            )

            vae = create_vae(
                strategy=strategy,
                pipeline_name=pipeline_name,
                **vae_init_kwargs,
            )

            if device is not None and dtype is not None:
                vae = vae.to(device=device, dtype=dtype)

            components._vae_cache[cache_key] = vae

        components.add("vae", vae)
