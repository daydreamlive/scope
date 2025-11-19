"""Base interface for all pipelines."""

from abc import ABC, abstractmethod

import torch
from pydantic import BaseModel


class Requirements(BaseModel):
    """Requirements for pipeline configuration.

    A pipeline's optional ``prepare`` method returns a ``Requirements``
    instance to describe how many input frames it needs for the next call.
    If ``prepare`` returns ``None`` the pipeline will be called without
    video input and is expected to operate purely in text / latent space.
    """

    # Number of frames the pipeline expects for the next call. The frame
    # processor will only invoke the pipeline once this many frames are
    # available in the buffer.
    input_size: int


class Pipeline(ABC):
    """Abstract base class for all pipelines.

    Contract:
    - ``__call__`` should accept an optional ``input`` argument representing
      a BCTHW tensor or a list of THWC tensors (in [0, 255]) when video
      input is required, or ``None`` when running in text-only mode.
    - Additional keyword arguments (e.g. ``prompts``, ``generation_mode``,
      ``init_cache``) are pipeline-specific but should be treated as
      stateless hints by the caller; pipelines manage their own internal
      state via ``self.state``.
    - The return value must be a THWC tensor in [0, 1] range representing
      the processed video chunk.
    """

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> dict:
        """Return default parameters for this pipeline.

        Implementations must return a dictionary with the following structure:

            {
                "native_generation_mode": "text" | "video",
                "modes": {
                    "text": {
                        "denoising_steps": [...],
                        "resolution": {"height": ..., "width": ...},
                        "manage_cache": ...,
                        "base_seed": ...,
                        "...": ...
                    },
                    "video": {
                        "denoising_steps": [...],
                        "resolution": {"height": ..., "width": ...},
                        "noise_scale": ...,
                        "noise_controller": ...,
                        "manage_cache": ...,
                        "base_seed": ...,
                        "...": ...
                    }
                }
            }

        Additional mode-specific keys (e.g. kv_cache_attention_bias) may be
        included as needed. All pipelines should provide entries for both text
        and video modes, even if a mode is not typically used.
        """
        pass

    @classmethod
    def _initialize_state_with_defaults(cls, state, config) -> None:
        """Initialize pipeline state with default values from config and pipeline defaults.

        This helper consolidates the common pattern of initializing state parameters
        with values from config (if present) or falling back to pipeline defaults.

        Args:
            state: PipelineState object to initialize
            config: Configuration object with optional overrides
        """
        from lib.defaults import get_mode_defaults

        native_defaults = get_mode_defaults(cls)

        state.set("current_start_frame", 0)
        state.set("current_noise_scale", 0.7)

        # Resolution
        default_resolution = native_defaults.get("resolution", {})
        state.set(
            "height",
            getattr(config, "height", default_resolution.get("height", 512)),
        )
        state.set(
            "width",
            getattr(config, "width", default_resolution.get("width", 576)),
        )

        # Seed
        state.set(
            "base_seed", getattr(config, "seed", native_defaults.get("base_seed", 42))
        )

        # Cache management
        state.set("manage_cache", native_defaults.get("manage_cache", True))

        # Noise controls
        state.set("noise_scale", native_defaults.get("noise_scale"))
        state.set("noise_controller", native_defaults.get("noise_controller"))

        # Pipeline-specific parameters (optional, may not exist in all defaults)
        kv_cache_bias = native_defaults.get("kv_cache_attention_bias")
        if kv_cache_bias is not None:
            state.set("kv_cache_attention_bias", kv_cache_bias)

    @abstractmethod
    def __call__(
        self, input: torch.Tensor | list[torch.Tensor] | None = None, **kwargs
    ) -> torch.Tensor:
        """Process a chunk of video frames."""
        pass
