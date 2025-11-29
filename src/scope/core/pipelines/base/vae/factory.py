"""VAE factory with strategy pattern for flexible VAE selection."""

from typing import Any

# Registry mapping strategy names to VAE class module paths
_VAE_REGISTRY: dict[str, tuple[str, str] | type] = {
    "krea_realtime_video": (
        "scope.core.pipelines.base.vae.krea_realtime_video",
        "KreaRealtimeVideoVAE",
    ),
    "longlive": ("scope.core.pipelines.base.vae.longlive", "LongLiveVAE"),
    "streamdiffusionv2": (
        "scope.core.pipelines.base.vae.streamdiffusionv2",
        "StreamDiffusionV2VAE",
    ),
    "streamdiffusionv2_longlive_scaled": (
        "scope.core.pipelines.base.vae.streamdiffusionv2",
        "StreamDiffusionV2VAEWithLongLiveScaling",
    ),
    "lightvae": (
        "scope.core.pipelines.base.vae.lightvae",
        "LightVAEWrapper",
    ),
    "streaming_lightvae": (
        "scope.core.pipelines.base.vae.lightvae",
        "StreamingLightVAEWrapper",
    ),
    "streaming_lightvae_longlive_scaled": (
        "scope.core.pipelines.base.vae.lightvae",
        "StreamingLightVAEWrapperWithLongLiveScaling",
    ),
}

# Default strategy for each pipeline
_PIPELINE_DEFAULTS: dict[str, str] = {
    "krea_realtime_video": "krea_realtime_video",
    "longlive": "longlive",
    "streamdiffusionv2": "streamdiffusionv2",
}


def _get_vae_class(strategy: str):
    """Lazy import of VAE class to avoid circular imports."""
    registry_entry = _VAE_REGISTRY[strategy]
    if isinstance(registry_entry, tuple):
        module_path, class_name = registry_entry
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        # Direct class registration
        return registry_entry


def register_vae_strategy(name: str, vae_class: type) -> None:
    """Register a new VAE strategy.

    Args:
        name: Strategy name for registry lookup
        vae_class: VAE class to register
    """
    _VAE_REGISTRY[name] = vae_class


def get_available_strategies() -> list[str]:
    """Get list of available VAE strategy names."""
    return list(_VAE_REGISTRY.keys())


def create_vae(
    strategy: str | None = None,
    pipeline_name: str | None = None,
    **kwargs: Any,
):
    """Create a VAE wrapper instance using strategy pattern.

    Args:
        strategy: VAE strategy name. If None, uses the default for the pipeline.
        pipeline_name: Pipeline name to determine default strategy if strategy is None.
        **kwargs: Additional arguments passed to the VAE constructor.

    Returns:
        VAE wrapper instance.

    Raises:
        ValueError: If strategy is unknown or both strategy and pipeline_name are None.

    Examples:
        # Use default strategy for pipeline
        vae = create_vae(pipeline_name="streamdiffusionv2", model_dir="wan_models")

        # Use explicit strategy
        vae = create_vae(strategy="longlive", model_dir="wan_models")

        # Register custom VAE
        register_vae_strategy("custom", CustomVAE)
        vae = create_vae(strategy="custom", custom_param=value)
    """
    # Determine which strategy to use
    if strategy is None:
        if pipeline_name is None:
            raise ValueError(
                "Either strategy or pipeline_name must be provided when strategy is None"
            )
        strategy = _PIPELINE_DEFAULTS.get(pipeline_name)
        if strategy is None:
            raise ValueError(
                f"Unknown pipeline name: {pipeline_name}. "
                f"Available pipelines: {list(_PIPELINE_DEFAULTS.keys())}"
            )

    # Get the VAE class (lazy import)
    registry_entry = _VAE_REGISTRY.get(strategy)
    if registry_entry is None:
        raise ValueError(
            f"Unknown VAE strategy: {strategy}. "
            f"Available strategies: {get_available_strategies()}"
        )

    # Handle both tuple (module path) and direct class registration
    if isinstance(registry_entry, tuple):
        vae_class = _get_vae_class(strategy)
    else:
        vae_class = registry_entry

    # Instantiate and return
    return vae_class(**kwargs)
