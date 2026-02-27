"""Base interface for all pipelines."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from .schema import BasePipelineConfig


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


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

    Subclasses declare transient event keys via the ``events`` class variable.
    Event keys are consumed once per ``__call__()`` and auto-cleared from
    ``PipelineState`` via ``_clear_events()``.  Persistent parameters (prompts,
    noise_scale, height, …) are unaffected.
    """

    # Keys that are transient events — consumed once per __call__() and
    # auto-cleared from PipelineState afterwards.  Subclasses extend this set.
    # NOTE: ``transition`` is intentionally excluded because transitions span
    # multiple chunks and are managed by PipelineProcessor with state-aware
    # clearing logic.
    events: ClassVar[frozenset[str]] = frozenset()

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

    def _clear_events(self, state, provided_kwargs: dict) -> None:
        """Clear event keys from state that were not provided in this call.

        This prevents stale transient values (e.g. video frames, VACE
        conditioning, lora_scales) from leaking into subsequent chunks.

        Args:
            state: The PipelineState instance to clear events from.
            provided_kwargs: The kwargs dict passed to the current __call__().
        """
        for key in self.events:
            if key not in provided_kwargs:
                state.set(key, None)

    @abstractmethod
    def __call__(self, **kwargs) -> dict:
        """
        Process a chunk of video frames.

        Args:
            **kwargs: Pipeline parameters. The input video is passed with the "video" key.
                The video value is a list of tensors, where each tensor has shape
                (1, H, W, C) in THWC format with values in [0, 255] range (uint8).
                The list contains one tensor per frame. Other common parameters include
                prompts, init_cache, etc.

        Returns:
            A dictionary containing the processed video tensor under the "video" key.
            The video tensor is in THWC format and [0, 1] range.
        """
        pass

