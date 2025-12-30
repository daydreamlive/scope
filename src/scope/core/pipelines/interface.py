"""Base interface for all pipelines."""

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from .schema import BasePipelineConfig


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


class Pipeline(ABC):
    """Abstract base class for all pipelines.

    Pipelines automatically get their config class from schema.yaml in their directory.
    This enables:
    - Validation via model_validate() / model_validate_json()
    - JSON Schema generation via model_json_schema()
    - Type-safe configuration access
    - API introspection and automatic UI generation

    To create a new pipeline:
    1. Create a directory for your pipeline (e.g., my_pipeline/)
    2. Add a schema.yaml with pipeline metadata and defaults
    3. Create pipeline.py with your Pipeline subclass

    See schema.py for the BasePipelineConfig model and available fields.
    For multi-mode pipeline support (text/video), pipelines use helper functions
    from defaults.py (resolve_input_mode, apply_mode_defaults_to_state, etc.).
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the Pydantic config class for this pipeline.

        Automatically loads from schema.yaml in the same directory as the
        pipeline subclass. No need to override this method - just provide
        a schema.yaml file.

        The config class defines:
        - pipeline_id: Unique identifier
        - pipeline_name: Human-readable name
        - pipeline_description: Capabilities description
        - Default parameter values for the pipeline

        Returns:
            Pydantic config model class loaded from schema.yaml
        """
        from .schema_loader import load_config_from_yaml

        # Find the directory containing this pipeline subclass
        module = inspect.getmodule(cls)
        if module is None or module.__file__ is None:
            # Fallback to base config if we can't find the module
            from .schema import BasePipelineConfig
            return BasePipelineConfig

        pipeline_dir = Path(module.__file__).parent
        schema_path = pipeline_dir / "schema.yaml"

        if schema_path.exists():
            return load_config_from_yaml(schema_path)

        # Fallback to base config if no schema.yaml found
        from .schema import BasePipelineConfig
        return BasePipelineConfig

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
