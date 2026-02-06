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

        if hasattr(self, "blocks"):
            try:
                del self.blocks
            except Exception:
                pass

        self._cleanup_gpu_objects(self, depth=0, max_depth=2, visited=set())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cleanup_gpu_objects(
        self, obj: object, depth: int, max_depth: int, visited: set
    ) -> None:
        """Recursively find and delete GPU objects (nn.Module, Tensor) from __dict__.

        Args:
            obj: Object to inspect
            depth: Current recursion depth
            max_depth: Maximum recursion depth (2 levels catches wrapper patterns)
            visited: Set of object ids already visited (prevents circular refs)
        """
        import torch

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if depth >= max_depth:
            return

        if not hasattr(obj, "__dict__"):
            return

        attrs_to_delete = []
        objects_to_recurse = []

        for attr_name, attr_value in list(obj.__dict__.items()):
            if isinstance(attr_value, (torch.nn.Module, torch.Tensor)):
                attrs_to_delete.append(attr_name)
            elif (
                hasattr(attr_value, "__dict__")
                and not isinstance(attr_value, type)
                and not isinstance(
                    attr_value, (str, bytes, int, float, bool, list, dict, tuple)
                )
            ):
                objects_to_recurse.append(attr_value)

        for attr_name in attrs_to_delete:
            try:
                delattr(obj, attr_name)
            except Exception:
                pass

        for nested_obj in objects_to_recurse:
            self._cleanup_gpu_objects(nested_obj, depth + 1, max_depth, visited)
