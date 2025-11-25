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
