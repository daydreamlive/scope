"""Base interface for all pipelines."""

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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

    def cleanup(self) -> None:
        """Release GPU resources. Called before pipeline is unloaded."""
        import torch

        if hasattr(self, "components"):
            try:
                components_dict = getattr(self.components, "_components", None)
                if components_dict is not None:
                    for name in list(components_dict.keys()):
                        del components_dict[name]
                del self.components
            except Exception:
                pass

        if hasattr(self, "state"):
            try:
                if hasattr(self.state, "values"):
                    self.state.values.clear()
                del self.state
            except Exception:
                pass

        for attr_name in list(self.__dict__.keys()):
            try:
                delattr(self, attr_name)
            except Exception:
                pass

        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch._dynamo.reset()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
