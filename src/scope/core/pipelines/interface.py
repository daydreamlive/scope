"""Base interface for all pipelines."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from diffusers.modular_pipelines import PipelineState
from pydantic import BaseModel, Field, field_validator


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


class PipelineModeConfig(BaseModel):
    """Configuration defaults for a single generation mode (text or video).

    Pipeline-specific parameters can be added via the model_config extra="allow" setting.
    Common optional parameters include:
    - vae_strategy: str - VAE strategy name for this mode (e.g. "longlive", "streamdiffusionv2")
    - input_size: int - Number of frames to request when this mode requires video input
    """

    denoising_steps: list[int] | None = Field(
        default=None, description="Default denoising steps for this mode"
    )
    resolution: dict[str, int] = Field(
        ..., description="Default resolution with height and width keys"
    )
    manage_cache: bool = Field(
        default=True, description="Default cache management setting"
    )
    base_seed: int = Field(default=42, description="Default random seed")
    noise_scale: float | None = Field(
        default=None, description="Default noise scale (None if not applicable)"
    )
    noise_controller: bool | None = Field(
        default=None,
        description="Default noise controller setting (None if not applicable)",
    )
    kv_cache_attention_bias: float | None = Field(
        default=None,
        description="Default KV cache attention bias (None if not applicable)",
    )

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate resolution has required height and width keys."""
        if "height" not in v or "width" not in v:
            raise ValueError("resolution must contain both 'height' and 'width' keys")
        if not isinstance(v["height"], int) or not isinstance(v["width"], int):
            raise ValueError("resolution height and width must be integers")
        if v["height"] <= 0 or v["width"] <= 0:
            raise ValueError("resolution height and width must be positive")
        return v

    model_config = {"extra": "allow"}


class PipelineDefaults(BaseModel):
    """Mode-aware defaults for pipelines supporting text and video generation modes.

    Pipelines declare their native mode and provide mode-specific configurations.
    """

    native_generation_mode: str = Field(
        ...,
        description="Native generation mode for this pipeline (text or video)",
    )
    modes: dict[str, PipelineModeConfig] = Field(
        ...,
        description="Mode-specific defaults. Keys are 'text' and 'video'",
    )

    model_config = {"extra": "forbid"}


class Pipeline(ABC):
    """Abstract base class for all pipelines.

    Note: Use GENERATION_MODE_TEXT and GENERATION_MODE_VIDEO constants from defaults.py
    when overriding NATIVE_GENERATION_MODE in subclasses.

    Architecture Note - ISP Consideration:
    All pipelines are currently required to declare generation mode support (text/video)
    even if they only actively use one mode. This is an acknowledged Interface Segregation
    Principle deviation, accepted because:
    1. All pipelines will support both modes in future iterations
    2. The temporary violation simplifies the architecture during the transition
    3. The overhead is minimal (declaration only, no forced implementation)
    """

    NATIVE_GENERATION_MODE: str = "text"
    PIPELINE_REGISTRY_NAME: str | None = None

    @classmethod
    def _create_mode_configs(
        cls,
        shared: dict[str, Any],
        text_overrides: dict[str, Any] | None = None,
        video_overrides: dict[str, Any] | None = None,
    ) -> dict[str, PipelineModeConfig]:
        """Helper to create mode configs with shared values and mode-specific overrides.

        Args:
            shared: Common configuration values shared across both modes
            text_overrides: Text-mode specific overrides (default: empty)
            video_overrides: Video-mode specific overrides (default: empty)

        Returns:
            Dictionary with 'text' and 'video' mode configurations
        """
        text_overrides = text_overrides or {}
        video_overrides = video_overrides or {}

        from .defaults import GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO

        return {
            GENERATION_MODE_TEXT: PipelineModeConfig(**{**shared, **text_overrides}),
            GENERATION_MODE_VIDEO: PipelineModeConfig(**{**shared, **video_overrides}),
        }

    @classmethod
    def _build_defaults(
        cls,
        shared: dict[str, Any],
        text_overrides: dict[str, Any] | None = None,
        video_overrides: dict[str, Any] | None = None,
    ) -> PipelineDefaults:
        """Build PipelineDefaults with native mode and mode configs.

        This helper reduces boilerplate by combining _create_mode_configs with
        PipelineDefaults construction using the class's NATIVE_GENERATION_MODE.

        Args:
            shared: Common configuration values shared across both modes
            text_overrides: Text-mode specific overrides (default: empty)
            video_overrides: Video-mode specific overrides (default: empty)

        Returns:
            PipelineDefaults object ready to return from get_defaults()

        Example:
            return cls._build_defaults(
                shared={"denoising_steps": [750, 500], "manage_cache": True, "base_seed": 42},
                text_overrides={"resolution": {"height": 512, "width": 512}},
                video_overrides={"resolution": {"height": 320, "width": 576}},
            )
        """
        return PipelineDefaults(
            native_generation_mode=cls.NATIVE_GENERATION_MODE,
            modes=cls._create_mode_configs(shared, text_overrides, video_overrides),
        )

    @classmethod
    def _initialize_with_native_mode_defaults(
        cls, state: PipelineState, config: Any
    ) -> None:
        """Initialize state with native mode defaults.

        This helper consolidates the pattern of getting native mode config
        and calling _initialize_state_with_defaults.

        Args:
            state: PipelineState object to initialize
            config: Configuration object with optional overrides
        """
        from .defaults import get_mode_defaults

        native_mode_config = get_mode_defaults(cls)
        cls._initialize_state_with_defaults(state, config, native_mode_config)

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> PipelineDefaults:
        """Return mode-aware default parameters for this pipeline.

        Returns:
            PipelineDefaults object containing native_generation_mode
            and mode-specific configurations for both text and video modes.

        Example:
            return PipelineDefaults(
                native_generation_mode="video",
                modes={
                    "text": PipelineModeConfig(
                        denoising_steps=[1000, 750],
                        resolution={"height": 512, "width": 512},
                        manage_cache=True,
                        base_seed=42,
                        noise_scale=None,
                        noise_controller=None,
                    ),
                    "video": PipelineModeConfig(
                        denoising_steps=[750, 250],
                        resolution={"height": 512, "width": 512},
                        manage_cache=True,
                        base_seed=42,
                        noise_scale=0.7,
                        noise_controller=True,
                    ),
                },
            )
        """
        pass

    @classmethod
    def _initialize_state_with_defaults(
        cls, state: PipelineState, config: Any, mode_config: PipelineModeConfig
    ) -> None:
        """Initialize pipeline state with default values from config and mode-specific defaults.

        This helper consolidates the common pattern of initializing state parameters
        with values from config (if present) or falling back to mode-specific defaults.

        Args:
            state: PipelineState object to initialize
            config: Configuration object with optional overrides
            mode_config: PipelineModeConfig object with mode-specific default values
        """
        # Common state initialization
        state.set("current_start_frame", 0)

        # Resolution from config or defaults (resolution is required in PipelineModeConfig)
        state.set(
            "height",
            getattr(config, "height", mode_config.resolution["height"]),
        )
        state.set(
            "width",
            getattr(config, "width", mode_config.resolution["width"]),
        )

        # Seed from config or defaults
        state.set("base_seed", getattr(config, "seed", mode_config.base_seed))

        # Cache management from defaults
        state.set("manage_cache", mode_config.manage_cache)

        # Noise controls from defaults
        if mode_config.noise_scale is not None:
            state.set("noise_scale", mode_config.noise_scale)

        if mode_config.noise_controller is not None:
            state.set("noise_controller", mode_config.noise_controller)

        # Pipeline-specific parameters (optional, may not exist in all defaults)
        if mode_config.kv_cache_attention_bias is not None:
            state.set("kv_cache_attention_bias", mode_config.kv_cache_attention_bias)

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
        from .defaults import GENERATION_MODE_VIDEO, get_mode_defaults

        mode = (
            generation_mode
            or kwargs.get("generation_mode")
            or self.NATIVE_GENERATION_MODE
        )

        if mode == GENERATION_MODE_VIDEO:
            mode_config = get_mode_defaults(self.__class__, mode)
            input_size = getattr(mode_config, "input_size", None)
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
        from .defaults import get_mode_defaults

        mode_config = get_mode_defaults(self.__class__, mode)

        strategy_key = f"{component_name}_strategy"
        strategy = getattr(mode_config, strategy_key, None)

        if cache_attr:
            if not hasattr(self, cache_attr):
                setattr(self, cache_attr, {})
            cache = getattr(self, cache_attr)

            cache_key = (mode, strategy)
            if cache_key in cache:
                return cache[cache_key]

        pipeline_name = self.PIPELINE_REGISTRY_NAME
        if pipeline_name is None:
            raise ValueError(
                f"{self.__class__.__name__} must define PIPELINE_REGISTRY_NAME class attribute"
            )

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
        self, state: PipelineState, components: Any, **kwargs
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

        mode = state.get("generation_mode") or self.NATIVE_GENERATION_MODE

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
