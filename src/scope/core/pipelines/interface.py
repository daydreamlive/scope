"""Base interface for all pipelines.

A :class:`Pipeline` is a :class:`scope.core.nodes.BaseNode` subclass —
the "heavy" kind that batches video frames, loads GPU models, and
carries a rich Pydantic config class. The graph editor and user-facing
docs call them *Nodes*; the name ``Pipeline`` survives as the
implementation base class so existing subclasses and plugins keep
working unchanged.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from scope.core.nodes.base import BaseNode, NodeDefinition, NodePort

if TYPE_CHECKING:
    from .schema import BasePipelineConfig


class Requirements(BaseModel):
    """Requirements for pipeline configuration."""

    input_size: int


class Pipeline(BaseNode, ABC):
    """Abstract base class for video-pipeline nodes.

    Subclasses implement ``__call__`` (the per-chunk processing
    function) and ``get_config_class`` (returning a Pydantic config
    that drives validation, JSON-schema generation, the parameter
    panel, and parameter defaults). Everything else — registry,
    plugin hook, graph editor — is the same as for plain nodes.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the Pydantic config class for this pipeline.

        Subclasses override to return their concrete config; the
        default returns ``BasePipelineConfig``.
        """
        from .schema import BasePipelineConfig

        return BasePipelineConfig

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        """Project the pipeline's config class into a :class:`NodeDefinition`.

        Populates the compact node-catalog fields (id, ports, etc.)
        and stuffs the full ``get_schema_with_metadata()`` output into
        ``pipeline_meta``, which is the rich data ``PipelineNode.tsx``
        renders in the parameter panel. ``params`` is left empty
        because the Pydantic schema is too structured to flatten into
        ``NodeParam[]`` widgets.
        """
        config = cls.get_config_class()
        return NodeDefinition(
            node_type_id=config.pipeline_id,
            display_name=getattr(config, "pipeline_name", config.pipeline_id),
            category="pipeline",
            description=getattr(config, "pipeline_description", "") or "",
            inputs=[
                NodePort(name=name, port_type="video")
                for name in (getattr(config, "inputs", ["video"]) or ["video"])
            ],
            outputs=[
                NodePort(name=name, port_type="video")
                for name in (getattr(config, "outputs", ["video"]) or ["video"])
            ],
            params=[],
            continuous=False,
            pipeline_meta=config.get_schema_with_metadata(),
        )

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
