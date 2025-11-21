"""Base interface for all pipelines."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from diffusers.modular_pipelines import PipelineState
from pydantic import BaseModel, Field, field_validator


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


class PipelineDefaults(BaseModel):
    """Typed defaults for pipeline configuration.

    Each pipeline must provide these core defaults. Pipeline-specific parameters
    can be added via the model_config extra="allow" setting.
    """

    denoising_steps: list[int] | None = Field(
        default=None, description="Default denoising steps for the pipeline"
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


class Pipeline(ABC):
    """Abstract base class for all pipelines."""

    @classmethod
    @abstractmethod
    def get_defaults(cls) -> PipelineDefaults:
        """Return typed default parameters for this pipeline.

        Returns:
            PipelineDefaults object with pipeline-specific configuration.

        Example:
            return PipelineDefaults(
                denoising_steps=[700, 500],
                resolution={"height": 512, "width": 512},
                manage_cache=True,
                base_seed=42,
                noise_scale=0.7,
                noise_controller=True,
                kv_cache_attention_bias=0.3
            )
        """
        pass

    @classmethod
    def _initialize_state_with_defaults(
        cls, state: PipelineState, config: Any, defaults: PipelineDefaults
    ) -> None:
        """Initialize pipeline state with default values from config and pipeline defaults.

        This helper consolidates the common pattern of initializing state parameters
        with values from config (if present) or falling back to pipeline defaults.

        Args:
            state: PipelineState object to initialize
            config: Configuration object with optional overrides
            defaults: PipelineDefaults object with default values
        """
        # Common state initialization
        state.set("current_start_frame", 0)

        # Resolution from config or defaults (resolution is required in PipelineDefaults)
        state.set(
            "height",
            getattr(config, "height", defaults.resolution["height"]),
        )
        state.set(
            "width",
            getattr(config, "width", defaults.resolution["width"]),
        )

        # Seed from config or defaults
        state.set("base_seed", getattr(config, "seed", defaults.base_seed))

        # Cache management from defaults
        state.set("manage_cache", defaults.manage_cache)

        # Noise controls from defaults
        if defaults.noise_scale is not None:
            state.set("noise_scale", defaults.noise_scale)

        if defaults.noise_controller is not None:
            state.set("noise_controller", defaults.noise_controller)

        # Pipeline-specific parameters (optional, may not exist in all defaults)
        if defaults.kv_cache_attention_bias is not None:
            state.set("kv_cache_attention_bias", defaults.kv_cache_attention_bias)

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
