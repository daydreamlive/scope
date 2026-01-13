"""Base interface for all pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from .schema import BasePipelineConfig


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


@dataclass
class PipelineOutput:
    """Output from pipeline inference.

    Pipelines can return either a plain torch.Tensor (legacy behavior) or a
    PipelineOutput instance for richer output including audio.

    Attributes:
        video: Video tensor in THWC format with values in [0, 1] range.
        audio: Optional audio tensor in (channels, samples) format, float32.
               Typically stereo (2 channels) at 24kHz sample rate.
        audio_sample_rate: Sample rate of the audio in Hz (e.g., 24000).
    """

    video: torch.Tensor
    audio: torch.Tensor | None = None
    audio_sample_rate: int | None = None


class Pipeline(ABC):
    """Abstract base class for all pipelines.

    Pipelines must implement get_config_class() to return their Pydantic config model.
    This enables:
    - Validation via model_validate() / model_validate_json()
    - JSON Schema generation via model_json_schema()
    - Type-safe configuration access
    - API introspection and automatic UI generation

    See schema.py for the BasePipelineConfig model and pipeline-specific configs.
    For multi-mode pipeline support (text/video), pipelines use helper functions
    from defaults.py (resolve_input_mode, apply_mode_defaults_to_state, etc.).
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the Pydantic config class for this pipeline.

        The config class should inherit from BasePipelineConfig and define:
        - pipeline_id: Unique identifier
        - pipeline_name: Human-readable name
        - pipeline_description: Capabilities description
        - pipeline_version: Version string
        - Default parameter values for the pipeline

        Returns:
            Pydantic config model class

        Note:
            Subclasses should override this method to return their config class.
            The default implementation returns BasePipelineConfig.

        Example:
            from .schema import LongLiveConfig

            @classmethod
            def get_config_class(cls) -> type[BasePipelineConfig]:
                return LongLiveConfig
        """
        from .schema import BasePipelineConfig

        return BasePipelineConfig

    @abstractmethod
    def __call__(
        self, input: torch.Tensor | list[torch.Tensor] | None = None, **kwargs
    ) -> torch.Tensor | PipelineOutput:
        """
        Process a chunk of video frames.

        Args:
            input: A tensor in BCTHW format OR a list of frame tensors in THWC format (in [0, 255] range), or None
            **kwargs: Additional parameters

        Returns:
            A processed chunk tensor in THWC format and [0, 1] range,
            OR a PipelineOutput instance containing video and optional audio.
        """
        pass
